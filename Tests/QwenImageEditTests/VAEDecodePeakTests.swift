// Isolate the VAE decode's peak at 1024² (latent 128×128) — is it the inference-peak
// driver worth a tiled decode, and how much does bf16 already save? Measures the peak of
// a single decode for fp32 vs bf16 weights.
//
// Run: QIE_VAEPEAK=1 swift test --filter VAEDecodePeakTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class VAEDecodePeakTests: XCTestCase {
    static let vaeDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511/vae")

    func testDecodePeak() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_VAEPEAK"] == "1", "QIE_VAEPEAK=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.vaeDir.path), "missing \(Self.vaeDir.path)")
        let gb = 1_000_000_000.0

        for dt in [DType.float32, .bfloat16] {
            let vae = try QwenImageEditWeights.loadVAE(directory: Self.vaeDir, dtype: dt)
            // 1024² output -> latent 128×128, 16 channels, 1 temporal frame.
            let latent = MLXRandom.normal([1, 16, 1, 128, 128]).asType(.float32)
            eval(vae.parameters())
            MLX.GPU.clearCache()
            MLX.GPU.resetPeakMemory()
            let before = Double(MLX.GPU.activeMemory) / gb
            let out = vae.decode(QwenImageVAE.deNormalize(latent))
            eval(out)
            let peak = Double(MLX.GPU.peakMemory) / gb
            print(String(format: "[vaepeak] %@ VAE: resident %.1f GB -> decode peak %.1f GB (+%.1f)",
                "\(dt)", before, peak, peak - before))
        }
    }
}
