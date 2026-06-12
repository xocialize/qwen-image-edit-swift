// S4 pre-gate: VAE encode vs the P2 golden conditioning latents.
//
// Golden: latents.safetensors cond_latents (1, 4096, 64) — the fox image LANCZOS'd
// to the 1024²-area VAE size, [-1,1]-normalized, encoded (argmax mode), latent-
// normalized, and 2x2-packed by the diffusers pipeline.
//
// Run: QIE_PARITY=1 swift test --filter VAEEncodeParityTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class VAEEncodeParityTests: XCTestCase {
    func testEncodeGolden() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PARITY"] == "1", "QIE_PARITY=1")

        let goldens = URL(
            fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
        let modelDir = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")

        let gold = try MLX.loadArrays(url: goldens.appendingPathComponent("latents.safetensors"))
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: goldens.appendingPathComponent("goldens_meta.json")))
            as! [String: Any]
        let vaeSize = meta["vae_size"] as! [Int]  // [w, h]
        let image = try EncoderParityTests.loadRGB(
            url: URL(fileURLWithPath: meta["input_image"] as! String))

        // diffusers VaeImageProcessor.preprocess: LANCZOS resize + [0,1] -> [-1,1]
        let (vw, vh) = (vaeSize[0], vaeSize[1])
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
        let pixels = MLXArray(chw, [1, 3, 1, vh, vw])

        let vae = try QwenImageEditWeights.loadVAE(
            directory: modelDir.appendingPathComponent("vae"), dtype: .float32)
        let latents = vae.encode(pixels)  // (1, 16, 1, vh/8, vw/8)
        let packed = QwenImagePipeline.packLatents(latents)
        let ref = gold["cond_latents"]!
        XCTAssertEqual(packed.shape, ref.shape, "packed shape")

        let a = packed.asType(.float32).flattened()
        let b = ref.flattened()
        let cos = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)) + 1e-12)).item(Float.self)
        let mse = mean(square(a - b)).item(Float.self)
        print("VAE encode: cosine \(cos)  mse \(mse)")
        XCTAssertGreaterThanOrEqual(cos, 0.999)
    }
}
