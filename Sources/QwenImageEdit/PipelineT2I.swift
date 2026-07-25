// Qwen-Image text-to-image generation pipeline — Swift mirror of diffusers
// `QwenImagePipeline` (pipeline_qwenimage.py), the reference for both the base
// Qwen/Qwen-Image weights and NVIDIA's 4-step DMD2 distill `nvidia/Qwen-Image-Flash`.
//
// Shares the whole stack with the edit path (same 60-layer DiT — weight-key sets are
// IDENTICAL across Qwen-Image / Qwen-Image-Edit-2511 / Qwen-Image-Flash — same Qwen2.5-VL
// encoder, same 3D causal VAE). The T2I deltas vs `QwenImageEditGenerator`, all from the
// reference:
//   1. no conditioning image: no VAE-encode branch, `imgShapes` is the target grid alone,
//      and the DiT's `zero_cond_t` per-token modulation index collapses to nil (every image
//      token takes the real-t params = the reference `zero_cond_t=false` branch);
//   2. text-only prompt encoding: T2I template, drop_idx 34 (edit uses 64), 512 tokens;
//   3. explicit width/height instead of an input image's aspect;
//   4. the STATIC time shift (`use_dynamic_shifting: false`), so `calculateShift`/mu are
//      unused — Flash packages shift 3.0, giving [1.0, 0.9, 0.75, 0.5, 0.0] over 4 steps.
//
// Flash internalized CFG 4.0 during distillation, so it runs `trueCFGScale = 1` — one DiT
// forward per step, no negative branch. The same generator runs the base Qwen-Image weights
// at 50 steps / true CFG 4.0.

import Foundation
import MLX
import MLXProfiling
import MLXRandom

/// End-to-end text-to-image: VL text encode -> denoise -> decode.
///
/// Per-stage residency matches `QwenImageEditGenerator`: the ~16 GB Qwen2.5-VL encoder is
/// loaded per request, used once, and evicted before the denoise peak, so only the DiT and
/// the small VAE are resident through the long phase. See that type's doc comment for the
/// rationale and the `keepEncoderResident` escape hatch.
public final class QwenImageT2IGenerator {
    /// Lazy loader for the prompt encoder. Text-only (`loadTextOnly`) is enough here.
    public let encoderProvider: () async throws -> QwenVLPromptEncoder
    public let transformer: QwenImageTransformer2DModel
    public let vae: QwenImageVAE
    public let keepEncoderResident: Bool
    /// Scheduler `shift` from the snapshot's scheduler_config.json (static shifting).
    public let shift: Float

    private var residentEncoder: QwenVLPromptEncoder?

    /// Single-entry prompt-embedding memo — a re-roll (new seed/steps/size, same prompt)
    /// skips the per-request encoder load entirely. Dropped with the generator on unload().
    private struct PromptCacheKey: Equatable {
        var prompt: String
        /// nil when the request runs CFG-free (no negative branch encoded).
        var negativePrompt: String?
    }
    private var cachedPromptEmbeds: (key: PromptCacheKey, pos: MLXArray, neg: MLXArray?)?

    /// Diagnostics from the last generate(): denoise steps skipped by the step-residual
    /// cache per CFG branch (nil when the cache was off).
    public private(set) var lastStepCacheSkips: (pos: Int, neg: Int)?

    public init(
        encoderProvider: @escaping () async throws -> QwenVLPromptEncoder,
        transformer: QwenImageTransformer2DModel,
        vae: QwenImageVAE,
        shift: Float = 3.0,
        keepEncoderResident: Bool = false
    ) {
        self.encoderProvider = encoderProvider
        self.transformer = transformer
        self.vae = vae
        self.shift = shift
        self.keepEncoderResident = keepEncoderResident
    }

    private func loadEncoder(isolation: isolated (any Actor)? = #isolation) async throws
        -> QwenVLPromptEncoder
    {
        if keepEncoderResident, let residentEncoder { return residentEncoder }
        let encoder = try await encoderProvider()
        if keepEncoderResident { residentEncoder = encoder }
        return encoder
    }

    private func evictEncoder(_ encoder: inout QwenVLPromptEncoder?) {
        guard !keepEncoderResident else { return }
        encoder = nil
        Memory.clearCache()
    }

    /// diffusers rounds the requested size down to a multiple of `vae_scale_factor * 2`
    /// (= 16); the model card says "use width and height divisible by 16 to avoid automatic
    /// resizing".
    public static func roundedSize(width: Int, height: Int) -> (width: Int, height: Int) {
        (max(width / 16, 1) * 16, max(height / 16, 1) * 16)
    }

    /// Build the first-forward graphs at load time so the first user request doesn't pay
    /// kernel/graph build: one small DiT step on the single-grid (nil modulateIndex) shape
    /// family plus a VAE decode, at 256². The VL encoder is deliberately NOT warmed — it
    /// loads per request, so warming it would pay the ~16 GB transient for nothing.
    public func warmup(isolation: isolated (any Actor)? = #isolation) {
        let dtype: DType = .bfloat16
        let side = 256
        let latentSide = side / 8
        let prof = MLXProfiler.shared
        let span = prof.begin("warmup", "first-forward", note: "256² T2I DiT step + VAE decode")

        let latents = QwenImagePipeline.packLatents(
            MLXArray.zeros([1, 16, 1, latentSide, latentSide]).asType(dtype))
        let txt = MLXArray.zeros([1, 32, 3584]).asType(dtype)
        let noise = transformer(
            hiddenStates: latents, encoderHiddenStates: txt,
            encoderHiddenStatesMask: nil, timestep: MLXArray([Float(1)]),
            imgShapes: [(1, latentSide / 2, latentSide / 2)])
        let unpacked = QwenImagePipeline.unpackLatents(
            noise.asType(.float32), pixelHeight: side, pixelWidth: side)
        let decoded = vae.decode(QwenImageVAE.deNormalize(unpacked))
        eval(decoded)
        prof.end(span)
        Memory.clearCache()
    }

    /// Generate an image from `prompt`. Returns interleaved RGB8 + dimensions.
    ///
    /// Defaults are Flash's: 4 steps, `trueCFGScale` 1.0 (guidance was internalized by the
    /// DMD2 distillation — applying CFG again double-counts it), 1024×1024. `latents` lets
    /// a parity gate inject the reference's noise instead of MLX's RNG, whose seed stream is
    /// not compatible with torch's.
    public func generate(
        prompt: String,
        negativePrompt: String = " ",
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 4,
        trueCFGScale: Float = 1.0,
        seed: UInt64 = 0,
        latents injectedLatents: MLXArray? = nil,
        stepCache: StepCacheMode = .off,
        maxSequenceLength: Int = QwenVLPromptEncoder.t2iMaxSequenceLength,
        progress: ((Int, Int) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> (pixels: [UInt8], width: Int, height: Int) {
        guard steps > 0 else {
            throw QwenImageEditError.invalidInput("steps must be > 0")
        }
        let (tw, th) = Self.roundedSize(width: width, height: height)
        guard tw >= 16, th >= 16 else {
            throw QwenImageEditError.invalidInput("size \(width)×\(height) is below 16×16")
        }

        // 1. Prompt encoding. The negative branch exists only under true CFG (> 1); Flash
        //    runs at 1.0, so the encode + a whole DiT forward per step are skipped —
        //    matching the reference `do_true_cfg`.
        let doCFG = trueCFGScale > 1
        let prof = MLXProfiler.shared
        let promptKey = PromptCacheKey(
            prompt: prompt, negativePrompt: doCFG ? negativePrompt : nil)
        let posEmbeds: MLXArray
        let negEmbeds: MLXArray?
        if let cached = cachedPromptEmbeds, cached.key == promptKey {
            posEmbeds = cached.pos
            negEmbeds = cached.neg
        } else {
            let eSpan = prof.begin(
                "encode", "vl",
                note: "per-request VL-7B load + text encode\(doCFG ? " ×2 (CFG)" : "")")
            var encoderRef: QwenVLPromptEncoder? = try await loadEncoder()
            posEmbeds = try encoderRef!.encodeText(
                prompt: prompt, maxSequenceLength: maxSequenceLength)
            negEmbeds = doCFG
                ? try encoderRef!.encodeText(
                    prompt: negativePrompt, maxSequenceLength: maxSequenceLength)
                : nil
            // Materialize off the encoder graph, then drop the encoder before the DiT peak.
            if let negEmbeds { eval(posEmbeds, negEmbeds) } else { eval(posEmbeds) }
            prof.end(eSpan)
            evictEncoder(&encoderRef)
            cachedPromptEmbeds = (promptKey, posEmbeds, negEmbeds)
        }
        let dtype = posEmbeds.dtype
        // CAN seam: prompt encoding done (encoder evicted), before the denoise loop.
        try Task.checkCancellation()

        // 2. Seeded noise, packed. num_channels_latents = in_channels // 4 = 16.
        var latents: MLXArray
        if let injectedLatents {
            latents = injectedLatents.asType(dtype)
        } else {
            let key = MLXRandom.key(seed)
            latents = QwenImagePipeline.packLatents(
                MLXRandom.normal([1, 16, 1, th / 8, tw / 8], key: key).asType(dtype))
        }

        // 3. Scheduler — static shift (dynamic shifting is off for this model family).
        let sigmas = QwenImagePipeline.staticShiftedSigmas(steps: steps, shift: shift)
        let imgShapes = [(1, th / 16, tw / 16)]

        // 4. Denoise loop.
        let posStepCache = stepCache.threshold.map { DiTStepCache(threshold: $0) }
        let negStepCache = doCFG ? stepCache.threshold.map { DiTStepCache(threshold: $0) } : nil
        for i in 0..<steps {
            // CAN cadence: one cooperative checkpoint per denoise step.
            try Task.checkCancellation()
            let span = prof.begin(
                "denoise", "step", index: i, note: String(format: "σ=%.3f", sigmas[i]))
            let t = MLXArray([sigmas[i]])
            let pos = transformer(
                hiddenStates: latents, encoderHiddenStates: posEmbeds,
                encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes,
                stepCache: posStepCache)
            let noise: MLXArray
            if let negEmbeds {
                let neg = transformer(
                    hiddenStates: latents, encoderHiddenStates: negEmbeds,
                    encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes,
                    stepCache: negStepCache)
                noise = QwenImagePipeline.guidedNoise(pos: pos, neg: neg, scale: trueCFGScale)
            } else {
                noise = pos
            }
            latents = latents + (sigmas[i + 1] - sigmas[i]) * noise
            eval(latents)
            prof.end(span)
            progress?(i + 1, steps)
        }
        lastStepCacheSkips = (posStepCache != nil || negStepCache != nil)
            ? (posStepCache?.skippedSteps ?? 0, negStepCache?.skippedSteps ?? 0) : nil

        // CAN seam: denoise done, before the monolithic VAE decode (one MLX eval).
        try Task.checkCancellation()

        // 5. Decode.
        let dSpan = prof.begin("vae-decode", "decode")
        let unpacked = QwenImagePipeline.unpackLatents(
            latents.asType(.float32), pixelHeight: th, pixelWidth: tw)
        let decoded = vae.decode(QwenImageVAE.deNormalize(unpacked))  // (1,3,1,H,W)
        let img = clip((decoded.squeezed(axis: 2) + 1) * 127.5, min: 0, max: 255)
            .asType(.uint8)
        let hwc = img[0].transposed(1, 2, 0)
        eval(hwc)
        prof.end(dSpan)
        return (hwc.asArray(UInt8.self), tw, th)
    }
}
