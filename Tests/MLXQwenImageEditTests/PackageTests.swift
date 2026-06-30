// Engine-conformance smoke: manifest + load -> run(IEditRequest) -> PNG -> unload.
//
// Run: QIE_PKG=1 swift test --filter PackageTests

import Foundation
import MLX
import MLXToolKit
import QwenImageEdit
import XCTest

@testable import MLXQwenImageEdit

final class PackageTests: XCTestCase {
    func testManifest() {
        let m = QwenImageEditPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .imageEdit)
        XCTAssertEqual(m.license.weightLicense, .apache2)
    }

    func testLoadRunUnload() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PKG"] == "1", "QIE_PKG=1")

        let package = QwenImageEditPackage(configuration: .init())
        try await package.load()

        let foxData = try Data(
            contentsOf: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Desktop/lens-t2i-package.png"))
        let request = IEditRequest(
            images: [Image(format: .png, data: foxData)],
            prompt: "Give the fox a small red scarf around its neck. Keep everything else unchanged.",
            steps: 8,
            guidanceScale: 4.0,
            seed: 7)
        let start = Date()
        let response = try await package.run(request)
        guard let edit = response as? IEditResponse else {
            return XCTFail("wrong response type")
        }
        print("package edit: \(edit.image.width ?? 0)x\(edit.image.height ?? 0) in \(Date().timeIntervalSince(start))s")
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qwen-edit-package-demo.png")
        try edit.image.data.write(to: out)
        print("saved \(out.path)")

        await package.unload()
    }

    /// Split-footprint mem-bench for the efficiency contract (1.14.0). Measures the per-stage
    /// residency the P2 encoder-eviction refactor produces, so `residentBytes` /
    /// `peakActivationBytes` are declared from data, not a guess. Synthetic input (no external
    /// file) at the 1024²-area envelope. Reports:
    ///   - resident floor   = DiT + VAE after clearCache (encoder NOT held — the new persistent set)
    ///   - encode-phase peak = floor + transient VL-7B encoder weights + encode activations
    ///   - denoise-phase peak = floor + DiT activation high-water (encoder already evicted)
    /// peakActivationBytes = max(encode-phase, denoise-phase) − floor.
    ///
    /// Prewarm the snapshot first to keep the cold 40 GB DiT load out of a live command buffer
    /// (watchdog), e.g. `cat <snapshot>/transformer/*.safetensors > /dev/null` before running.
    /// Run: QIE_MEMBENCH=1 swift test --filter PackageTests/testSplitFootprintMemBench
    func testSplitFootprintMemBench() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_MEMBENCH"] == "1", "QIE_MEMBENCH=1")
        let snapshot = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: snapshot.path), "missing \(snapshot.path)")
        let gb = 1_000_000_000.0
        func active() -> Double { Double(MLX.GPU.activeMemory) / gb }
        func peak() -> Double { Double(MLX.GPU.peakMemory) / gb }

        // Persistent set: DiT (bf16) + VAE (fp32). Encoder is loaded per request via the provider.
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        eval(transformer.parameters())
        eval(vae.parameters())
        MLX.GPU.clearCache()
        let floor = active()
        print(String(format: "[membench] resident floor (DiT+VAE, encoder evicted): %.1f GB", floor))

        let generator = QwenImageEditGenerator(
            encoderProvider: { try await QwenVLPromptEncoder.load(snapshot: snapshot) },
            transformer: transformer, vae: vae)  // keepEncoderResident defaults false

        // Synthetic 1024² input at the envelope.
        let side = 1024
        let rgb = [UInt8](repeating: 128, count: side * side * 3)

        MLX.GPU.clearCache()
        MLX.GPU.resetPeakMemory()
        let (_, w, h) = try await generator.generate(
            image: (rgb: rgb, width: side, height: side),
            prompt: "Add a small red scarf. Keep everything else unchanged.",
            negativePrompt: " ", steps: 8, trueCFGScale: 4.0, seed: 7, progress: { _, _ in })
        let worstPeak = peak()
        let activation = max(0, worstPeak - floor)
        print(String(
            format: "[membench] %dx%d 8-step CFG4 | worst peak %.1f GB | floor %.1f GB | "
                + "activation (peak-floor) %.1f GB", w, h, worstPeak, floor, activation))
        print(String(
            format: "[membench] DECLARE -> residentBytes=%.0f  peakActivationBytes=%.0f (+20%% headroom -> %.0f)",
            floor * gb, activation * gb, activation * 1.2 * gb))
    }
}
