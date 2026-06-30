// Strength-tuning sweep: render a few general-purpose stylistic effect LoRAs at several
// strengths over the Lightning 4-step base, to decide registry defaultStrength (the
// ecosystem default is 1.0; the question is whether our Lightning-4step regime under-applies
// effect LoRAs the way it under-applied Lightning itself). Reuses ONE resident int4 DiT and
// hot-swaps [Lightning@4.0, effect@s] per combo. Writes qie-sweep-<effect>-s<n>.png to ~/Desktop.
//
// CONCLUSION (2026-06-27, pixar/anime/noir on the fox @ s=1/2/3): effect LoRAs are correctly
// applied at strength 1.0 and do NOT benefit from >1.0 — pixar and noir visibly OVERCOOK at
// s=3 (blur, color drift, lost fidelity); anime is flat across strengths. This is the opposite
// of the Lightning *distillation* LoRA (deltas below bf16 ULP → needed 4.0): these are
// full-strength adapters trained to run at 1.0 (matching the reference HF space). So registry
// defaultStrength stays 1.0; per-request metaData[loraStrength] remains the power-user override.
//
// Run: QIE_SWEEP=1 swift test --filter LoRAStrengthSweepTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class LoRAStrengthSweepTests: XCTestCase {
    func testStrengthSweep() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIE_SWEEP"] == "1", "QIE_SWEEP=1")
        typealias C = LoRACommunityRenderTests
        let dir = C.loraDir
        let effects: [(tag: String, file: URL, prompt: String)] = [
            ("pixar", dir.appendingPathComponent("Pixar-Inspired-3D.safetensors"),
             "Transform it into Pixar-inspired 3D"),
            ("anime", dir.appendingPathComponent("Photo-to-Anime.safetensors"),
             "transform into anime"),
            ("noir", dir.appendingPathComponent("Noir-Comic-Book.safetensors"),
             "Transform into a noir comic book style"),
        ]
        let strengths: [Float] = [1.0, 2.0, 3.0]

        for p in [C.quantizedDiT, C.quantizedEnc, C.lightning.path, C.input.path] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }
        for e in effects {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: e.file.path), "missing \(e.file.path)")
        }

        let image = try EncoderParityTests.loadRGB(url: C.input)
        let desktop = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop")
        let encoder = try await QwenVLPromptEncoder.load(
            snapshot: C.modelDir, quantizedTextModelPath: C.quantizedEnc)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: C.modelDir.appendingPathComponent("vae"), dtype: .bfloat16)

        // One resident DiT; hot-swap [Lightning, effect@s] per combo.
        let dit = try QwenImageEditWeights.loadQuantizedDiT(from: URL(fileURLWithPath: C.quantizedDiT))
        let swapper = QwenImageEditLoRASwapper(model: dit)
        let gen = QwenImageEditGenerator(encoder: encoder, transformer: dit, vae: vae)

        for e in effects {
            for s in strengths {
                try swapper.set([(C.lightning, 4.0), (e.file, s)])
                let t = Date()
                let (px, w, h) = try await gen.generate(
                    image: image, prompt: e.prompt, negativePrompt: " ",
                    steps: 4, trueCFGScale: 1.0, seed: 42, progress: { _, _ in })
                let out = desktop.appendingPathComponent("qie-sweep-\(e.tag)-s\(Int(s)).png")
                try GenerateDemoTests.writePNG(pixels: px, width: w, height: h, to: out)
                print("[sweep] \(e.tag)@\(s): \(w)x\(h) in "
                    + "\(String(format: "%.1f", Date().timeIntervalSince(t)))s -> \(out.lastPathComponent)")
            }
            GPU.clearCache()
        }
    }
}
