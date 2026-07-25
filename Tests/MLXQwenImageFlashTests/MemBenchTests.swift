// Split-footprint mem-bench for the Flash T2I tier (efficiency contract 1.14.0) — the
// measurement that replaces the footprint currently inherited from the edit package.
//
// Same shape as MLXQwenImageEditTests/PackageTests.testSplitFootprintMemBench, at the T2I
// envelope: 4 steps, 1024², true CFG 1.0 (one DiT forward per step, no negative branch).
// T2I has no conditioning tokens, so the DiT sequence is the target grid alone — the
// activation term should come in UNDER the edit path's.
//
// Prewarm the snapshot first to keep the cold 40 GB DiT load out of a live command buffer
// (watchdog): `cat <snapshot>/transformer/*.safetensors > /dev/null`.
// Run: QIF_MEMBENCH=1 swift test --filter MemBenchTests

import Foundation
import MLX
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashMemBenchTests: XCTestCase {

    func testSplitFootprintMemBench() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_MEMBENCH"] == "1", "QIF_MEMBENCH=1")
        let snapshot = URL(fileURLWithPath: FlashPackageTests.snapshot)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: snapshot.path), "missing \(snapshot.path)")
        let gb = 1_000_000_000.0
        func active() -> Double { Double(MLX.GPU.activeMemory) / gb }
        func peak() -> Double { Double(MLX.GPU.peakMemory) / gb }

        // Persistent set: DiT (bf16) + VAE (fp32). The VL encoder is a per-request transient.
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        eval(transformer.parameters())
        eval(vae.parameters())
        MLX.GPU.clearCache()
        let floor = active()
        print(String(format: "[membench] resident floor (DiT+VAE): %.1f GB", floor))

        let generator = QwenImageT2IGenerator(
            encoderProvider: { try await QwenVLPromptEncoder.loadTextOnly(snapshot: snapshot) },
            transformer: transformer, vae: vae, shift: 3.0)

        MLX.GPU.clearCache()
        MLX.GPU.resetPeakMemory()
        let (_, w, h) = try await generator.generate(
            prompt: "A red fox in a snowy pine forest at golden hour, photorealistic",
            width: 1024, height: 1024, steps: 4, trueCFGScale: 1.0, seed: 42,
            progress: { _, _ in })
        let worstPeak = peak()
        let activation = max(0, worstPeak - floor)
        print(String(
            format: "[membench] %dx%d 4-step cfg1 | worst peak %.1f GB | floor %.1f GB | "
                + "activation (peak-floor) %.1f GB", w, h, worstPeak, floor, activation))
        print(String(
            format: "[membench] DECLARE -> residentBytes=%.0f  peakActivationBytes=%.0f "
                + "(+20%% headroom -> %.0f)",
            floor * gb, activation * gb, activation * 1.2 * gb))
    }
}
