// Live LoRA gate for the T2I tier: do community Qwen-Image adapters actually map onto the
// Flash DiT, and does a stacked render still come out coherent?
//
// This is the real risk in shipping a LoRA gallery. Flash is a DMD2 student of Qwen/Qwen-Image
// with the architecture intact (60 layers, 24 heads, joint_attention_dim 3584), so adapters
// trained on the teacher load by shape — but "by shape" is a claim about the checkpoint, not
// about our dialect remapper. `QwenImageEditLoRA` targets KEYS PRESENT IN THE FILE, so a
// dialect it doesn't recognize silently matches nothing: the render succeeds, looks like the
// base model, and nobody notices. `activeKeys` is the discriminator, which is why the sweep
// asserts on it per adapter rather than eyeballing one image.
//
// ⚠ GPU stream only (quantized DiT).
//
// Run: QIF_LORA_LIVE=1 swift test --filter FlashLoRALiveTests

import Foundation
import MLX
import MLXToolKit
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashLoRALiveTests: XCTestCase {

    /// The local mirror staged by the gallery script (Workstream A).
    static let mirror = URL(
        fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/xocialize/qwen-image-flash-loras")
    static let quantSnapshot = FlashQuantGateTests.quantSnapshot
    static let goldens = FlashQuantGateTests.goldens

    private func requireLive() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_LORA_LIVE"] == "1", "QIF_LORA_LIVE=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.quantSnapshot.path),
            "missing \(Self.quantSnapshot.path)")
    }

    /// Attach every mirrored adapter in turn and assert it actually binds parameters. Reports a
    /// per-adapter table so an incompatible one can be dropped from the published gallery rather
    /// than shipping as a no-op.
    func testEveryMirroredAdapterBinds() throws {
        try requireLive()
        let loraDir = Self.mirror.appendingPathComponent("loras")
        let files = ((try? FileManager.default.contentsOfDirectory(
            at: loraDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try XCTSkipUnless(!files.isEmpty, "no mirrored adapters at \(loraDir.path)")

        let transformer = try QwenImageEditWeights.loadQuantizedDiT(
            from: Self.quantSnapshot.appendingPathComponent(
                QwenImageFlashConfiguration.int8DiTFile))
        let swapper = QwenImageEditLoRASwapper(model: transformer)

        var bound: [String] = []
        var unbound: [String] = []
        for file in files {
            let name = file.deletingPathExtension().lastPathComponent
            do {
                try swapper.set([(file, 1.0)])
                let keys = swapper.activeKeys.count
                if keys > 0 {
                    bound.append("\(name): \(keys) keys")
                } else {
                    unbound.append("\(name): 0 keys (dialect not recognized)")
                }
            } catch {
                unbound.append("\(name): \(error)")
            }
            swapper.detach()
        }

        print("[lora-sweep] bound \(bound.count)/\(files.count)")
        for line in bound { print("   ✓ \(line)") }
        for line in unbound { print("   ✗ \(line)") }
        XCTAssertTrue(
            unbound.isEmpty,
            "adapters that bind nothing would render as the base model with no error: \(unbound)")
    }

    /// One adapter, one render, at the production grid — the visual half of the gate.
    /// `QIF_LORA_ID` selects which mirrored adapter (default: the first alphabetically).
    func testStyledRender1024() async throws {
        try requireLive()
        let loraDir = Self.mirror.appendingPathComponent("loras")
        let chosen = ProcessInfo.processInfo.environment["QIF_LORA_ID"]
        let files = ((try? FileManager.default.contentsOfDirectory(
            at: loraDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard let file = chosen.map({ loraDir.appendingPathComponent("\($0).safetensors") })
            ?? files.first, FileManager.default.fileExists(atPath: file.path)
        else { throw XCTSkip("no mirrored adapter to render with") }
        let name = file.deletingPathExtension().lastPathComponent

        let transformer = try QwenImageEditWeights.loadQuantizedDiT(
            from: Self.quantSnapshot.appendingPathComponent(
                QwenImageFlashConfiguration.int8DiTFile))
        let vae = try QwenImageEditWeights.loadVAE(
            directory: Self.quantSnapshot.appendingPathComponent("vae"), dtype: .float32)
        let swapper = QwenImageEditLoRASwapper(model: transformer)
        try swapper.set([(file, 1.0)])
        XCTAssertGreaterThan(swapper.activeKeys.count, 0, "\(name) bound nothing")

        let snapshot = Self.quantSnapshot
        let encPath = snapshot.appendingPathComponent(
            QwenImageFlashConfiguration.int8TextEncoderFile).path
        let generator = QwenImageT2IGenerator(
            encoderProvider: {
                try await QwenVLPromptEncoder.loadTextOnly(
                    snapshot: snapshot, quantizedTextModelPath: encPath)
            },
            transformer: transformer, vae: vae, shift: 3.0)

        let start = Date()
        let (pixels, w, h) = try await generator.generate(
            prompt: "a lighthouse on a rocky coast at sunset",
            width: 1024, height: 1024, steps: 4, trueCFGScale: 1.0, seed: 42,
            progress: { _, _ in })
        print(String(
            format: "[lora-render] %@ (%d keys) %dx%d in %.1f s",
            name, swapper.activeKeys.count, w, h, -start.timeIntervalSinceNow))

        // The package's own PNG encoder — same path the engine returns to a consumer.
        let out = Self.goldens.appendingPathComponent("swift_t2i_lora_\(name)_1024.png")
        try QwenImageFlashPackage.encodePNG(pixels: pixels, width: w, height: h).write(to: out)
        print("[saved] \(out.path)")

        let arr = MLXArray(pixels).asType(.float32)
        let sd = sqrt(mean(square(arr - mean(arr))))
        eval(sd)
        XCTAssertGreaterThan(sd.item(Float.self), 10.0, "styled output looks degenerate")
    }
}
