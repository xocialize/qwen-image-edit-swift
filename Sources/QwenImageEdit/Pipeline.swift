// Qwen-Image-Edit-2511 generation pipeline — Swift mirror of diffusers
// QwenImageEditPlusPipeline (the reference; see the deviation ledger in
// qwen-image-edit-mlx/PORTING-SPEC.md for where mflux differs).

import Foundation
import MLX
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
public final class QwenImageEditGenerator {
    public let encoder: QwenVLPromptEncoder
    public let transformer: QwenImageTransformer2DModel
    public let vae: QwenImageVAE

    public init(
        encoder: QwenVLPromptEncoder, transformer: QwenImageTransformer2DModel,
        vae: QwenImageVAE
    ) {
        self.encoder = encoder
        self.transformer = transformer
        self.vae = vae
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
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> (pixels: [UInt8], width: Int, height: Int) {
        try generate(
            images: [image], prompt: prompt, negativePrompt: negativePrompt,
            steps: steps, trueCFGScale: trueCFGScale, seed: seed, progress: progress)
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
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> (pixels: [UInt8], width: Int, height: Int) {
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
        let doCFG = trueCFGScale > 1
        let posEmbeds = try encoder.encode(prompt: prompt, images: images)
        let negEmbeds = doCFG ? try encoder.encode(prompt: negativePrompt, images: images) : nil
        let dtype = posEmbeds.dtype

        // 2. Per-image VAE conditioning latents (each at its own 1024²-area /32 size),
        //    concatenated on the sequence axis — reference cat(all_image_latents, dim=1).
        var condParts: [MLXArray] = []
        var condShapes: [(Int, Int, Int)] = []
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
        for i in 0..<steps {
            let t = MLXArray([sigmas[i]])
            let hidden = concatenated([latents, condLatents.asType(dtype)], axis: 1)
            let pos = transformer(
                hiddenStates: hidden, encoderHiddenStates: posEmbeds,
                encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes
            )[0..., ..<targetLen]
            let noise: MLXArray
            if let negEmbeds {
                let neg = transformer(
                    hiddenStates: hidden, encoderHiddenStates: negEmbeds,
                    encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes
                )[0..., ..<targetLen]
                noise = QwenImagePipeline.guidedNoise(
                    pos: pos, neg: neg, scale: trueCFGScale)
            } else {
                noise = pos
            }
            latents = latents + (sigmas[i + 1] - sigmas[i]) * noise
            eval(latents)
            progress?(i + 1, steps)
        }

        // 6. Decode.
        let unpacked = QwenImagePipeline.unpackLatents(
            latents.asType(.float32), pixelHeight: th, pixelWidth: tw)
        let decoded = vae.decode(QwenImageVAE.deNormalize(unpacked))  // (1,3,1,H,W)
        let img = clip((decoded.squeezed(axis: 2) + 1) * 127.5, min: 0, max: 255)
            .asType(.uint8)
        // (1,3,H,W) -> interleaved RGB8
        let hwc = img[0].transposed(1, 2, 0)
        eval(hwc)
        return (hwc.asArray(UInt8.self), tw, th)
    }
}
