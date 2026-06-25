// The true low-RAM tier: load the PRE-QUANTIZED DiT (no bf16 peak) + int4 encoder, and
// confirm the load peak — not just resident — drops well below the quantize-after-load
// path (~41 GB). Requires the converted file from QuantizedConvertTests.
//
// Run: QIE_PREQUANT=1 swift test --filter TurboPreQuantTests

import Foundation
import MLX
import MLXToolKit
import XCTest

@testable import MLXQwenImageEditTurbo

final class TurboPreQuantTests: XCTestCase {
    static let quantizedDiT =
        "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-dit-int4-mod8.safetensors"
    static let quantizedEncoder =
        "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-vl7b-int4.safetensors"

    func testPreQuantizedLoadPeak() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PREQUANT"] == "1", "QIE_PREQUANT=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.quantizedDiT),
            "missing \(Self.quantizedDiT) — run QuantizedConvertTests first")

        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.quantizedEncoder),
            "missing \(Self.quantizedEncoder) — run QuantizedConvertTests first")
        let config = QwenImageEditTurboConfiguration(
            quantizedDiTPath: Self.quantizedDiT, quantizedEncoderPath: Self.quantizedEncoder,
            lowPrecisionVAE: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: config.loraPath), "missing \(config.loraPath)")
        let inputPath = "/Users/dustinnielson/Desktop/lens-t2i-package.png"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: inputPath), "missing \(inputPath)")

        let gb = 1_000_000_000.0
        MLX.GPU.resetPeakMemory()  // measure THIS load, not anything prior in-process
        let pkg = QwenImageEditTurboPackage(configuration: config)
        try await pkg.load()
        let loadPeak = Double(MLX.GPU.peakMemory) / gb

        let data = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        let (_, w, h) = try QwenImageEditTurboPackage.decodeRGB(data)
        let request = IEditRequest(
            images: [Image(format: .png, data: data, width: w, height: h)],
            prompt: "Change the fox's fur color to snow white. Keep the pose, background and "
                + "lighting unchanged.",
            steps: 4, mode: turboMode)
        let edit = try await pkg.run(request) as! IEditResponse

        MLX.GPU.clearCache()
        let resident = Double(MLX.GPU.activeMemory) / gb
        let peak = Double(MLX.GPU.peakMemory) / gb
        print(String(format: "[prequant] load peak: %.1f GB  | resident: %.1f GB  | full peak: %.1f GB",
            loadPeak, resident, peak))

        XCTAssertEqual(edit.image.width, 1024)
        // Pre-quantized DiT AND encoder: no bf16 weights ever materialize, so load peak
        // should approach resident (~22 GB) — a true low-RAM tier, way under the 41 GB
        // quantize-after-load path.
        XCTAssertLessThan(loadPeak, 26.0, "full pre-quantized load peak unexpectedly high")
        // bf16 VAE keeps the 1024² decode intermediates small -> inference peak well under
        // the ~29 GB fp32-VAE path.
        XCTAssertLessThan(peak, 27.0, "inference peak unexpectedly high")
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qie-turbo-prequant.png")
        try edit.image.data.write(to: out)
        print("[prequant] wrote -> \(out.path)")
    }
}
