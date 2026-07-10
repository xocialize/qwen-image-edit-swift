// Qwen-Image-Edit-2511 generation pipeline — Swift mirror of diffusers
// QwenImageEditPlusPipeline (the reference; see the deviation ledger in
// qwen-image-edit-mlx/PORTING-SPEC.md for where mflux differs).

import Foundation
import MLX
import MLXProfiling
import MLXRandom

public enum QwenImagePipeline {

    // MARK: latent packing (diffusers _pack_latents / _unpack_latents)

    /// (B, 16, 1, H, W) -> (B, H/2*W/2, 64)
    public static func packLatents(_ x5: MLXArray) -> MLXArray {
        let x = x5.squeezed(axis: 2)
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, c, h / 2, 2, w / 2, 2)
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped(b, (h / 2) * (w / 2), c * 4)
    }

    /// (B, h/2*w/2, 64) -> (B, 16, 1, h, w) where h = pixelHeight/8.
    public static func unpackLatents(_ x: MLXArray, pixelHeight: Int, pixelWidth: Int)
        -> MLXArray
    {
        let b = x.dim(0)
        let h = pixelHeight / 8
        let w = pixelWidth / 8
        return x.reshaped(b, h / 2, w / 2, 16, 2, 2)
            .transposed(0, 3, 1, 4, 2, 5)
            .reshaped(b, 16, h, w)
            .expandedDimensions(axis: 2)
    }

    // MARK: scheduler (FlowMatchEuler, dynamic shifting, exponential time shift)

    /// diffusers calculate_shift — scheduler config: base 256/4096, shift 0.5/1.15.
    public static func calculateShift(
        imageSeqLen: Int, baseSeqLen: Int = 256, maxSeqLen: Int = 4096,
        baseShift: Float = 0.5, maxShift: Float = 1.15
    ) -> Float {
        let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
        let b = baseShift - m * Float(baseSeqLen)
        return Float(imageSeqLen) * m + b
    }

    /// sigmas linspace(1, 1/steps, steps) with exponential time shift; trailing 0.
    public static func shiftedSigmas(steps: Int, mu: Float) -> [Float] {
        let raw = (0..<steps).map { i -> Float in
            1.0 - Float(i) * (1.0 - 1.0 / Float(steps)) / Float(max(steps - 1, 1))
        }
        let eMu = exp(mu)
        var sigmas = raw.map { s in eMu / (eMu + (1.0 / s - 1.0)) }
        sigmas.append(0)
        return sigmas
    }

    /// Norm-rescaled true CFG (reference L831-835).
    public static func guidedNoise(pos: MLXArray, neg: MLXArray, scale: Float) -> MLXArray {
        let comb = neg + scale * (pos - neg)
        let condNorm = sqrt(sum(pos * pos, axis: -1, keepDims: true))
        let combNorm = sqrt(sum(comb * comb, axis: -1, keepDims: true))
        return comb * (condNorm / combNorm)
    }
}

/// End-to-end image editing: VL encode -> denoise (zero_cond_t DiT, true CFG) -> decode.
///
/// **Per-stage residency (efficiency contract 1.14.0).** The Qwen2.5-VL prompt encoder
/// (~16 GB bf16) is used ONCE per request to encode the prompt + conditioning images,
/// then sits idle through the entire DiT denoise loop and VAE decode — the heaviest,
/// longest phase. So the generator does NOT hold the encoder resident: it owns an async
/// `encoderProvider` (the wrapper's loader), loads the encoder on demand, encodes, then
/// **evicts it (`nil` + `Memory.clearCache()`) before the denoise peak**. Only the DiT
/// (transformer, with any bound LoRAs) and the small VAE stay resident — the DiT is the
/// resident floor + the activation peak. All three wrappers (base/Turbo/TeleStyle) share
/// this core, so they all inherit the eviction. Tradeoff: the encoder re-loads per request
/// (cheap encode vs. expensive denoise) — a `keepEncoderResident` flag covers big-RAM tiers.
public final class QwenImageEditGenerator {
    /// Lazy loader for the Qwen2.5-VL prompt encoder. Invoked per request, then evicted
    /// before the denoise peak (unless `keepEncoderResident`).
    public let encoderProvider: () async throws -> QwenVLPromptEncoder
    public let transformer: QwenImageTransformer2DModel
    public let vae: QwenImageVAE
    /// Keep the encoder resident across requests (skip per-request evict+reload). Default
    /// `false` = evict-between-stages, the memory-citizen default; set `true` on big-RAM tiers.
    public let keepEncoderResident: Bool

    /// Hot encoder when `keepEncoderResident` is set (avoids the reload each request).
    private var residentEncoder: QwenVLPromptEncoder?

    /// Single-entry VL prompt-embedding memo (FR6b — same shape as `cachedClonePrompt`
    /// in Qwen3TTSPackage). Iterative-editing UX (same conditioning images + prompts,
    /// new seed/steps/guidance) re-pays the VL encode — and, worse, the per-request
    /// ~16 GB encoder load — every re-roll. A hit skips `loadEncoder` entirely, so with
    /// evict-between-stages the 7B encoder never becomes resident for a re-roll. The
    /// cached embeddings are a few MB (1×S×3584 bf16). Dropped with the generator on
    /// package unload().
    private struct PromptCacheKey: Equatable {
        var imagesHash: Int
        var prompt: String
        /// nil when the request runs CFG-free (no negative branch encoded).
        var negativePrompt: String?
    }
    private var cachedPromptEmbeds: (key: PromptCacheKey, pos: MLXArray, neg: MLXArray?)?

    /// Diagnostics from the last generate(): denoise steps skipped by the step-residual
    /// cache per CFG branch (nil when the cache was off). For eval harnesses/telemetry.
    public private(set) var lastStepCacheSkips: (pos: Int, neg: Int)?

    private static func hashImages(_ images: [(rgb: [UInt8], width: Int, height: Int)])
        -> Int
    {
        var hasher = Hasher()
        for image in images {
            hasher.combine(image.width)
            hasher.combine(image.height)
            image.rgb.withUnsafeBytes { hasher.combine(bytes: $0) }
        }
        return hasher.finalize()
    }

    /// Staged init: the encoder is loaded on demand via `encoderProvider`, not held resident.
    public init(
        encoderProvider: @escaping () async throws -> QwenVLPromptEncoder,
        transformer: QwenImageTransformer2DModel,
        vae: QwenImageVAE,
        keepEncoderResident: Bool = false
    ) {
        self.encoderProvider = encoderProvider
        self.transformer = transformer
        self.vae = vae
        self.keepEncoderResident = keepEncoderResident
    }

    /// Back-compat init from an already-loaded encoder. The encoder is kept resident (the
    /// pre-staged behavior) since the caller has no way to reload it. Prefer the
    /// `encoderProvider` init to get per-stage eviction.
    public convenience init(
        encoder: QwenVLPromptEncoder, transformer: QwenImageTransformer2DModel,
        vae: QwenImageVAE
    ) {
        self.init(
            encoderProvider: { encoder }, transformer: transformer, vae: vae,
            keepEncoderResident: true)
    }

    /// Obtain the prompt encoder for this request. Reuses the hot encoder when
    /// `keepEncoderResident`, otherwise loads a fresh one via `encoderProvider`.
    private func loadEncoder(isolation: isolated (any Actor)? = #isolation) async throws
        -> QwenVLPromptEncoder
    {
        if keepEncoderResident, let residentEncoder { return residentEncoder }
        let encoder = try await encoderProvider()
        if keepEncoderResident { residentEncoder = encoder }
        return encoder
    }

    /// Drop the encoder's weights before the denoise peak. When keeping it resident this is a
    /// no-op (the hot encoder is held in `residentEncoder`); otherwise it nils the caller's
    /// last strong reference and clears the buffer cache, reclaiming the ~16 GB.
    private func evictEncoder(_ encoder: inout QwenVLPromptEncoder?) {
        guard !keepEncoderResident else { return }
        encoder = nil           // release the encoder's MLXArrays (last strong ref)
        Memory.clearCache()     // return the freed buffers to the OS before denoise
    }

    /// Build/execute the first-forward graphs at load time (FR4) so the first user edit
    /// doesn't pay kernel/graph build: one small DiT step with the real shape family
    /// (target grid + one conditioning grid, so the zero_cond_t modulateIndex path is
    /// warmed) plus a VAE encode + decode, all at 256². The VL encoder is deliberately
    /// NOT warmed — it loads per request (evict-between-stages), so a load-time warmup
    /// would pay the ~16 GB transient for nothing.
    public func warmup(isolation: isolated (any Actor)? = #isolation) {
        let dtype: DType = .bfloat16
        let side = 256
        let latentSide = side / 8
        let prof = MLXProfiler.shared
        let span = prof.begin(
            "warmup", "first-forward", note: "256² DiT step + VAE encode/decode at load")

        let condPixels = MLXArray.zeros([1, 3, 1, side, side]).asType(dtype)
        let cond = QwenImagePipeline.packLatents(vae.encode(condPixels))
        let latents = QwenImagePipeline.packLatents(
            MLXArray.zeros([1, 16, 1, latentSide, latentSide]).asType(dtype))
        let hidden = concatenated([latents, cond.asType(dtype)], axis: 1)
        let txt = MLXArray.zeros([1, 32, 3584]).asType(dtype)
        let grid = (1, latentSide / 2, latentSide / 2)
        let noise = transformer(
            hiddenStates: hidden, encoderHiddenStates: txt,
            encoderHiddenStatesMask: nil, timestep: MLXArray([Float(1)]),
            imgShapes: [grid, grid]
        )[0..., ..<latents.dim(1)]
        let unpacked = QwenImagePipeline.unpackLatents(
            noise.asType(.float32), pixelHeight: side, pixelWidth: side)
        let decoded = vae.decode(QwenImageVAE.deNormalize(unpacked))
        eval(decoded)
        prof.end(span)
        Memory.clearCache()  // return the warmup buffers before real requests
    }

    /// Edit a single `image` per `prompt`. Returns interleaved RGB8 + dimensions.
    /// Thin wrapper over the multi-image core (`generate(images:)`).
    public func generate(
        image: (rgb: [UInt8], width: Int, height: Int),
        prompt: String,
        negativePrompt: String = " ",
        steps: Int = 20,
        trueCFGScale: Float = 4.0,
        seed: UInt64 = 0,
        stepCache: StepCacheMode = .off,
        progress: ((Int, Int) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> (pixels: [UInt8], width: Int, height: Int) {
        try await generate(
            images: [image], prompt: prompt, negativePrompt: negativePrompt,
            steps: steps, trueCFGScale: trueCFGScale, seed: seed, stepCache: stepCache,
            progress: progress)
    }

    /// Multi-image edit / style transfer. Images are conditioning inputs in prompt
    /// order ("Picture 1", "Picture 2", …); for TeleStyleV2, image 0 = content,
    /// image 1 = style (the fused LoRA learned the order semantics — there is no
    /// role token). Mirrors the diffusers QwenImageEditPlusPipeline:
    ///   - VL branch consumes all images (handled inside `encoder.encode`);
    ///   - each image is VAE-encoded at its own 1024²-area /32 size, packed, and the
    ///     per-image cond latents are concatenated on the sequence axis;
    ///   - `imgShapes` = [output grid] + [one grid per conditioning image] (RoPE).
    /// Output size follows the first (content) image's aspect, matching the reference.
    public func generate(
        images: [(rgb: [UInt8], width: Int, height: Int)],
        prompt: String,
        negativePrompt: String = " ",
        steps: Int = 20,
        trueCFGScale: Float = 4.0,
        seed: UInt64 = 0,
        stepCache: StepCacheMode = .off,
        progress: ((Int, Int) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> (pixels: [UInt8], width: Int, height: Int) {
        guard let content = images.first else {
            throw QwenImageEditError.invalidInput("at least one image is required")
        }
        // Target (output) size: 1024²-area, ratio of the content image, /32.
        let (tw, th) = QwenVLPromptEncoder.calculateDimensions(
            targetArea: 1024 * 1024, ratio: Double(content.width) / Double(content.height))

        // 1. Prompt encoding (the VL branch resizes each image to 384²-area internally
        //    and emits "Picture i" blocks for all images). The negative branch is only
        //    needed when true CFG is active (scale > 1); the DMD/Lightning tier runs at
        //    scale 1.0 (no CFG), so we skip it — matching the reference `do_true_cfg`.
        //
        //    PER-STAGE EVICTION: load the VL-7B encoder, encode, force-materialize the
        //    embeddings (`eval`), then drop the encoder + clear the cache BEFORE the denoise
        //    loop so the ~16 GB encoder is not co-resident with the DiT activation peak.
        let doCFG = trueCFGScale > 1
        let prof = MLXProfiler.shared
        // Prompt-embedding memo (FR6b): a hit skips the per-request encoder load + encode
        // entirely — the 7B encoder never becomes resident for a re-roll.
        let promptKey = PromptCacheKey(
            imagesHash: Self.hashImages(images), prompt: prompt,
            negativePrompt: doCFG ? negativePrompt : nil)
        let posEmbeds: MLXArray
        let negEmbeds: MLXArray?
        if let cached = cachedPromptEmbeds, cached.key == promptKey {
            posEmbeds = cached.pos
            negEmbeds = cached.neg
        } else {
            // Manual span (MLX_PROFILE=1) across the await — includes the per-request encoder
            // load (evicted after; see per-stage eviction above); bounded by the `eval` below.
            let eSpan = prof.begin(
                "encode", "vl",
                note: "per-request VL-7B load + encode\(doCFG ? " ×2 (CFG)" : "")")
            // Scope the encoder so its only strong reference is released before the denoise
            // loop.
            var encoderRef: QwenVLPromptEncoder? = try await loadEncoder()
            posEmbeds = try encoderRef!.encode(prompt: prompt, images: images)
            negEmbeds =
                doCFG ? try encoderRef!.encode(prompt: negativePrompt, images: images) : nil
            // Materialize the embeddings off the encoder graph, drop the encoder ref, then
            // clear the cache — reclaiming the ~16 GB before the DiT denoise peak.
            if let negEmbeds { eval(posEmbeds, negEmbeds) } else { eval(posEmbeds) }
            prof.end(eSpan)
            evictEncoder(&encoderRef)
            cachedPromptEmbeds = (promptKey, posEmbeds, negEmbeds)
        }
        let dtype = posEmbeds.dtype
        // CAN seam: prompt encoding done (encoder evicted), before conditioning + denoise.
        try Task.checkCancellation()

        // 2. Per-image VAE conditioning latents (each at its own 1024²-area /32 size),
        //    concatenated on the sequence axis — reference cat(all_image_latents, dim=1).
        var condParts: [MLXArray] = []
        var condShapes: [(Int, Int, Int)] = []
        // Span note: the cond-latent graph has no eval of its own — its compute realizes at
        // denoise step 0's eval, so this row times CPU preprocessing + graph build only.
        let cSpan = prof.begin("vae-encode", "cond", note: "lazy — realizes at denoise step 0")
        for image in images {
            let (vw, vh) = QwenVLPromptEncoder.calculateDimensions(
                targetArea: 1024 * 1024, ratio: Double(image.width) / Double(image.height))
            let resized = PILLanczosResize.resize(
                rgb: image.rgb, width: image.width, height: image.height,
                outWidth: vw, outHeight: vh)
            var chw = [Float](repeating: 0, count: 3 * vh * vw)
            let plane = vh * vw
            for i in 0..<plane {
                let p = i * 3
                chw[i] = Float(resized[p]) / 255 * 2 - 1
                chw[plane + i] = Float(resized[p + 1]) / 255 * 2 - 1
                chw[2 * plane + i] = Float(resized[p + 2]) / 255 * 2 - 1
            }
            let condPixels = MLXArray(chw, [1, 3, 1, vh, vw]).asType(dtype)
            condParts.append(QwenImagePipeline.packLatents(vae.encode(condPixels)))
            condShapes.append((1, vh / 16, vw / 16))
        }
        let condLatents =
            condParts.count == 1 ? condParts[0] : concatenated(condParts, axis: 1)
        prof.end(cSpan)

        // 3. Seeded target noise, packed.
        let key = MLXRandom.key(seed)
        var latents = QwenImagePipeline.packLatents(
            MLXRandom.normal([1, 16, 1, th / 8, tw / 8], key: key).asType(dtype))

        // 4. Scheduler.
        let mu = QwenImagePipeline.calculateShift(imageSeqLen: latents.dim(1))
        let sigmas = QwenImagePipeline.shiftedSigmas(steps: steps, mu: mu)

        let imgShapes = [(1, th / 16, tw / 16)] + condShapes
        let targetLen = latents.dim(1)

        // 5. Denoise loop (true CFG when scale > 1, else single positive pass).
        // Step-residual caching (FR2, opt-in): pos and neg run as separate forwards, so
        // each CFG branch gets its OWN cache. Cache state lives only for this loop; the
        // retained arrays (prev shallow + deep residual per branch) are activation-sized
        // and counted under the declared peakActivationBytes headroom.
        let posStepCache = stepCache.threshold.map { DiTStepCache(threshold: $0) }
        let negStepCache = doCFG ? stepCache.threshold.map { DiTStepCache(threshold: $0) } : nil
        for i in 0..<steps {
            // CAN cadence: cooperative cancellation once per denoise step (throwing seam —
            // the CancellationError unwinds unchanged to the engine wrapper's run()).
            try Task.checkCancellation()
            // Manual span bounded by the loop's own per-step `eval(latents)` below.
            let span = prof.begin(
                "denoise", "step", index: i, note: String(format: "σ=%.3f", sigmas[i]))
            let t = MLXArray([sigmas[i]])
            let hidden = concatenated([latents, condLatents.asType(dtype)], axis: 1)
            let pos = transformer(
                hiddenStates: hidden, encoderHiddenStates: posEmbeds,
                encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes,
                stepCache: posStepCache
            )[0..., ..<targetLen]
            let noise: MLXArray
            if let negEmbeds {
                let neg = transformer(
                    hiddenStates: hidden, encoderHiddenStates: negEmbeds,
                    encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes,
                    stepCache: negStepCache
                )[0..., ..<targetLen]
                noise = QwenImagePipeline.guidedNoise(
                    pos: pos, neg: neg, scale: trueCFGScale)
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

        // 6. Decode. Manual span around the existing eval — `decode` is lazy; its compute
        // realizes at `eval(hwc)`.
        let dSpan = prof.begin("vae-decode", "decode")
        let unpacked = QwenImagePipeline.unpackLatents(
            latents.asType(.float32), pixelHeight: th, pixelWidth: tw)
        let decoded = vae.decode(QwenImageVAE.deNormalize(unpacked))  // (1,3,1,H,W)
        let img = clip((decoded.squeezed(axis: 2) + 1) * 127.5, min: 0, max: 255)
            .asType(.uint8)
        // (1,3,H,W) -> interleaved RGB8
        let hwc = img[0].transposed(1, 2, 0)
        eval(hwc)
        prof.end(dSpan)
        return (hwc.asArray(UInt8.self), tw, th)
    }
}
