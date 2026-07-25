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

        // QIF_LORA_DTYPE=bf16 loads the UNQUANTIZED DiT — the tier the app actually runs at
        // when the governor can seat bf16, and the one where the in-app LoRA render came out
        // black while the int8 CLI render was fine.
        let useBF16 = ProcessInfo.processInfo.environment["QIF_LORA_DTYPE"] == "bf16"
        let bf16Snapshot = URL(
            fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/nvidia/Qwen-Image-Flash")
        let snapshotRoot = useBF16 ? bf16Snapshot : Self.quantSnapshot
        let transformer = useBF16
            ? try QwenImageEditWeights.loadDiTFromPT(
                directory: bf16Snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
            : try QwenImageEditWeights.loadQuantizedDiT(
                from: Self.quantSnapshot.appendingPathComponent(
                    QwenImageFlashConfiguration.int8DiTFile))
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshotRoot.appendingPathComponent("vae"), dtype: .float32)
        let swapper = QwenImageEditLoRASwapper(model: transformer)
        try swapper.set([(file, 1.0)])
        XCTAssertGreaterThan(swapper.activeKeys.count, 0, "\(name) bound nothing")
        // Matrix lever: QIF_NO_CHAIN=1 disables the swapper-set per-block chaining, isolating
        // whether a failure is the long-graph race (chaining-sensitive) or something else.
        if ProcessInfo.processInfo.environment["QIF_NO_CHAIN"] == "1" {
            transformer.chainBlockGraphs = false
        }

        let snapshot = snapshotRoot
        let encPath = useBF16
            ? nil
            : snapshot.appendingPathComponent(
                QwenImageFlashConfiguration.int8TextEncoderFile).path
        let generator = QwenImageT2IGenerator(
            encoderProvider: {
                try await QwenVLPromptEncoder.loadTextOnly(
                    snapshot: snapshot, quantizedTextModelPath: encPath)
            },
            transformer: transformer, vae: vae, shift: 3.0)

        // QIF_LORA_SIZE bisects the mlx#3797 NAX window: at 512² the img FFN sees M=1024
        // (BELOW the 1366 threshold), at 1024² it sees M=4096 (the upper edge, in-window).
        let side = ProcessInfo.processInfo.environment["QIF_LORA_SIZE"].flatMap(Int.init) ?? 1024
        // QIF_LORA_W/H override for non-square repros (the demo renders 1360x768 = M 4080,
        // which manifested as BLACK/NaN in-app while square 1024^2 gave banded static).
        let width = ProcessInfo.processInfo.environment["QIF_LORA_W"].flatMap(Int.init) ?? side
        let height = ProcessInfo.processInfo.environment["QIF_LORA_H"].flatMap(Int.init) ?? side
        let start = Date()
        let (pixels, w, h) = try await generator.generate(
            prompt: "a lighthouse on a rocky coast at sunset",
            width: width, height: height, steps: 4, trueCFGScale: 1.0, seed: 42,
            progress: { _, _ in })
        print(String(
            format: "[lora-render] %@ (%d keys) %dx%d in %.1f s",
            name, swapper.activeKeys.count, w, h, -start.timeIntervalSinceNow))

        // The package's own PNG encoder — same path the engine returns to a consumer.
        let out = Self.goldens.appendingPathComponent(
            "swift_t2i_lora_\(name)_\(useBF16 ? "bf16" : "int8")_\(width)x\(height).png")
        try QwenImageFlashPackage.encodePNG(pixels: pixels, width: w, height: h).write(to: out)
        print("[saved] \(out.path)")

        assertLooksLikeAnImage(pixels, width: w, height: h, label: name)
    }

    /// Coherence check, not just a variance check.
    ///
    /// A plain `sd > 10` gate PASSES pure static — which is exactly how the first bf16+LoRA
    /// corruption slipped through: NAX-corrupted output is high-variance noise, so variance
    /// alone reads it as "not degenerate". Natural images are locally smooth, so the
    /// discriminator is neighbour difference: adjacent pixels correlate strongly in a real
    /// render and barely at all in noise.
    func assertLooksLikeAnImage(
        _ pixels: [UInt8], width: Int, height: Int, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let a = MLXArray(pixels).reshaped([height, width, 3]).asType(.float32)
        let sd = sqrt(mean(square(a - mean(a)))).item(Float.self)
        // Mean |horizontal neighbour difference|, normalized by the image's own contrast.
        let dx = abs(a[0..., 1...,  0...] - a[0..., ..<(width - 1), 0...])
        let roughness = (mean(dx).item(Float.self)) / max(sd, 1e-6)

        print(String(format: "[coherence] %@: sd %.1f  roughness %.3f", label, sd, roughness))
        XCTAssertGreaterThan(sd, 10.0, "\(label): flat/degenerate output", file: file, line: line)
        // Measured: coherent Flash renders sit near 0.1–0.3; NAX-corrupted static is ~1.0+.
        XCTAssertLessThan(
            roughness, 0.6,
            "\(label): output is high-frequency noise, not an image (NAX corruption signature)",
            file: file, line: line)
    }
}
