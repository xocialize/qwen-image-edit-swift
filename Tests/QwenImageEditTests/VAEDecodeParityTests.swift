// S2 gate: VAE decode vs the P2 golden (fp32; PSNR).
//
// Golden: vae_decode.safetensors — latent_in is ALREADY de-normalized (the capture
// applied latents_mean/std, matching the diffusers pipeline-side convention);
// decoded is diffusers fp32 CPU output (B, 3, 1, H, W).
//
// Run: QIE_PARITY=1 swift test --filter VAEDecodeParityTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class VAEDecodeParityTests: XCTestCase {
    func testDecodeGolden() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PARITY"] == "1",
            "set QIE_PARITY=1 to run")

        let goldens = URL(
            fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
        let modelDir = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")

        let gold = try MLX.loadArrays(
            url: goldens.appendingPathComponent("vae_decode.safetensors"))
        let vae = try QwenImageEditWeights.loadVAE(
            directory: modelDir.appendingPathComponent("vae"), dtype: .float32)

        let decoded = vae.decode(gold["latent_in"]!.asType(.float32))
        let ref = gold["decoded"]!
        XCTAssertEqual(decoded.shape, ref.shape, "shape mismatch")

        let diff = decoded - ref
        let mse = mean(diff * diff)
        // Reference pixel range is [-1, 1] -> peak 2.
        let psnr = 10 * log10(MLXArray(Float(4)) / mse)
        eval(psnr)
        let psnrV = psnr.item(Float.self)
        print("VAE decode PSNR: \(psnrV) dB")
        XCTAssertGreaterThanOrEqual(psnrV, 60, "fp32 decode should be near-exact")
    }
}
