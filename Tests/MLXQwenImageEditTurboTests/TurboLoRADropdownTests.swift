// End-to-end gate for the effect-LoRA dropdown surface: drive the turbo package through the
// ModelPackage contract and confirm that a community effect selected via metaData is
// resolved from the bundled registry, lazy-downloaded + cached from HuggingFace, hot-swapped
// onto the resident Lightning base, and rendered — all without reloading the 20B DiT.
//
// Run: QIE_TURBO_LORA=1 swift test --filter TurboLoRADropdownTests

import Foundation
import MLX
import MLXToolKit
import XCTest

@testable import MLXQwenImageEditTurbo

final class TurboLoRADropdownTests: XCTestCase {
    static let quantizedDiT =
        "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-dit-int4-mod8.safetensors"
    static let quantizedEncoder =
        "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-vl7b-int4.safetensors"
    static let inputPath = "/Users/dustinnielson/Desktop/lens-t2i-package.png"

    func testEffectDropdownEndToEnd() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TURBO_LORA"] == "1", "QIE_TURBO_LORA=1")
        for p in [Self.quantizedDiT, Self.quantizedEncoder, Self.inputPath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }

        // Route the lazy LoRA cache to a scratch dir so the download is observable + isolated.
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("qie-lora-dropdown-test-\(getpid())")
        let config = QwenImageEditTurboConfiguration(
            quantizedDiTPath: Self.quantizedDiT, quantizedEncoderPath: Self.quantizedEncoder,
            lowPrecisionVAE: true, modelsRootDirectory: cacheRoot)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: config.loraPath), "missing \(config.loraPath)")

        let data = try Data(contentsOf: URL(fileURLWithPath: Self.inputPath))
        let (_, w, h) = try QwenImageEditTurboPackage.decodeRGB(data)
        let img = Image(format: .png, data: data, width: w, height: h)
        let prompt = "Transform it into Pixar-inspired 3D"
        let desktop = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop")

        let pkg = QwenImageEditTurboPackage(configuration: config)
        try await pkg.load()

        // 1. Base turbo (no effect selected) — Lightning only.
        let base = try await pkg.run(
            IEditRequest(images: [img], prompt: prompt, steps: 4, mode: turboMode)) as! IEditResponse
        try base.image.data.write(to: desktop.appendingPathComponent("qie-dropdown-base.png"))

        // 2. Effect selected via metaData — registry lookup + HF download + hot-swap.
        let effectReq = IEditRequest(
            images: [img], prompt: prompt, steps: 4, mode: turboMode,
            metaData: [LoRAMetaKeys.id: .string("pixar-inspired-3d")])
        let effect = try await pkg.run(effectReq) as! IEditResponse
        try effect.image.data.write(to: desktop.appendingPathComponent("qie-dropdown-pixar.png"))

        // The selected adapter must have been downloaded into the cache.
        let cached = cacheRoot.appendingPathComponent("qie-lora-cache/pixar-inspired-3d.safetensors")
        XCTAssertTrue(FileManager.default.fileExists(atPath: cached.path),
            "effect LoRA was not lazy-cached at \(cached.path)")

        XCTAssertEqual(base.image.width, 1024)
        XCTAssertEqual(effect.image.width, 1024)
        // The effect must change the output (otherwise the swap didn't take).
        XCTAssertNotEqual(base.image.data, effect.image.data, "effect LoRA had no effect")

        // 3. Unknown id is a clean error, not a crash.
        await XCTAssertThrowsErrorAsync(
            _ = try await pkg.run(IEditRequest(
                images: [img], prompt: prompt, steps: 4, mode: turboMode,
                metaData: [LoRAMetaKeys.id: .string("does-not-exist")])))

        try? FileManager.default.removeItem(at: cacheRoot)
        print("[dropdown] base + pixar rendered; effect lazy-cached + hot-swapped OK")
    }
}

/// Minimal async throwing assertion (XCTest has no built-in async variant).
func XCTAssertThrowsErrorAsync(
    _ expression: @autoclosure () async throws -> Void,
    file: StaticString = #filePath, line: UInt = #line
) async {
    do { try await expression(); XCTFail("expected an error", file: file, line: line) }
    catch {}
}
