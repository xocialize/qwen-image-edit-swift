// Phase 4: int4 DiT tier validation. Loads the turbo package with ditBits=4 (quantize
// the DiT attn+mlp Linears, then apply the Lightning LoRA as QLoRALinear), measures the
// resident GPU footprint, and renders a 4-step edit to eyeball quality vs the bf16 tier.
//
// Run: QIE_TURBO_4BIT=1 swift test --filter Turbo4BitTests

import Foundation
import MLX
import MLXToolKit
import XCTest

@testable import MLXQwenImageEditTurbo

final class Turbo4BitTests: XCTestCase {
    func testInt4LoadAndEdit() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TURBO_4BIT"] == "1", "QIE_TURBO_4BIT=1")
        let config = QwenImageEditTurboConfiguration(
            ditBits: 4, encoderBits: 4, modulationBits: 8)
        for p in [config.snapshotPath, config.loraPath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }
        let inputPath = "/Users/dustinnielson/Desktop/lens-t2i-package.png"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: inputPath), "missing \(inputPath)")

        let gb = 1_000_000_000.0
        let pkg = QwenImageEditTurboPackage(configuration: config)
        try await pkg.load()

        let data = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        let (_, w, h) = try QwenImageEditTurboPackage.decodeRGB(data)
        let request = IEditRequest(
            images: [Image(format: .png, data: data, width: w, height: h)],
            prompt: "Change the fox's fur color to snow white. Keep the pose, background and "
                + "lighting unchanged.",
            steps: 4, mode: turboMode)  // strength + CFG come from config defaults
        let edit = try await pkg.run(request) as! IEditResponse

        // Measure AFTER the render: MLX is lazy, so quantization only materializes (and the
        // original bf16 weights free) once evaluated. clearCache() drops the reuse pool to
        // leave true resident.
        MLX.GPU.clearCache()
        let resident = Double(MLX.GPU.activeMemory) / gb
        let peak = Double(MLX.GPU.peakMemory) / gb
        print(String(format: "[turbo-4bit] resident: %.1f GB  (load+infer peak: %.1f GB)",
            resident, peak))

        XCTAssertEqual(edit.image.width, 1024)
        // Mixed: attn+mlp int4, modulation int8, encoder int4. Between the 28 GB
        // (mod full-precision) and 18 GB (mod int4) points; int8 mod trades some footprint
        // for quality.
        XCTAssertLessThan(resident, 26.0, "mixed int4/int8 resident unexpectedly high")
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qie-turbo-int4.png")
        try edit.image.data.write(to: out)
        print("[turbo-4bit] wrote \(edit.image.data.count) bytes -> \(out.path)")
    }
}
