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

    /// Edit `image` per `prompt`. Returns interleaved RGB8 + dimensions.
    public func generate(
        image: (rgb: [UInt8], width: Int, height: Int),
        prompt: String,
        negativePrompt: String = " ",
        steps: Int = 20,
        trueCFGScale: Float = 4.0,
        seed: UInt64 = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let ratio = Double(image.width) / Double(image.height)
        // Target + VAE conditioning sizes: 1024²-area, ratio-preserved, /32.
        let (tw, th) = QwenVLPromptEncoder.calculateDimensions(
            targetArea: 1024 * 1024, ratio: ratio)
        let (vw, vh) = (tw, th)  // same formula and area for the conditioning branch

        // 1. Prompt encoding (the VL branch resizes to 384²-area internally).
        let posEmbeds = try encoder.encode(prompt: prompt, images: [image])
        let negEmbeds = try encoder.encode(prompt: negativePrompt, images: [image])

        // 2. Conditioning latents: LANCZOS to VAE size, [-1,1], encode, pack.
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
        let dtype = posEmbeds.dtype
        let condPixels = MLXArray(chw, [1, 3, 1, vh, vw]).asType(dtype)
        let condLatents = QwenImagePipeline.packLatents(vae.encode(condPixels))

        // 3. Seeded target noise, packed.
        let key = MLXRandom.key(seed)
        var latents = QwenImagePipeline.packLatents(
            MLXRandom.normal([1, 16, 1, th / 8, tw / 8], key: key).asType(dtype))

        // 4. Scheduler.
        let mu = QwenImagePipeline.calculateShift(imageSeqLen: latents.dim(1))
        let sigmas = QwenImagePipeline.shiftedSigmas(steps: steps, mu: mu)

        let imgShapes = [(1, th / 16, tw / 16), (1, vh / 16, vw / 16)]
        let targetLen = latents.dim(1)

        // 5. Denoise loop with true CFG.
        for i in 0..<steps {
            let t = MLXArray([sigmas[i]])
            let hidden = concatenated([latents, condLatents.asType(dtype)], axis: 1)
            let pos = transformer(
                hiddenStates: hidden, encoderHiddenStates: posEmbeds,
                encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes
            )[0..., ..<targetLen]
            let neg = transformer(
                hiddenStates: hidden, encoderHiddenStates: negEmbeds,
                encoderHiddenStatesMask: nil, timestep: t, imgShapes: imgShapes
            )[0..., ..<targetLen]
            let noise = QwenImagePipeline.guidedNoise(pos: pos, neg: neg, scale: trueCFGScale)
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
