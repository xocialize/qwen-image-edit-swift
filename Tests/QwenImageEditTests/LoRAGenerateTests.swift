// Phase 2 functional gate: prove the Lightning LoRA survives bf16 RUNTIME application
// (apply(), not fuse) on real activations — the thing synthetic input can't show.
//
// Renders the same edit three ways for an eyeball comparison on ~/Desktop:
//   1. base, 4 steps, CFG 4.0   — what rushing the base to 4 steps looks like (undercooked)
//   2. base + Lightning, 4 steps, CFG 1.0 (DMD) — the fast tier; should be coherent
//   3. base, 20 steps, CFG 4.0  — quality reference (only when QIE_REF=1)
//
// Run: QIE_TURBO=1 [QIE_REF=1] swift test --filter LoRAGenerateTests

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import QwenImageEdit

final class LoRAGenerateTests: XCTestCase {
    static let modelDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
    static let lora = URL(
        fileURLWithPath:
            "/Users/dustinnielson/Development/telestyle-work/loras/"
            + "QIE-2511-Lightning-4steps-V1.0-bf16.safetensors")

    func testLightning4StepRender() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TURBO"] == "1", "QIE_TURBO=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.lora.path), "missing \(Self.lora.path)")
        let wantRef = ProcessInfo.processInfo.environment["QIE_REF"] == "1"

        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json")))
            as! [String: Any]
        let image = try EncoderParityTests.loadRGB(
            url: URL(fileURLWithPath: meta["input_image"] as! String))
        let prompt = meta["prompt"] as! String
        let neg = meta["negative_prompt"] as! String
        let seed = UInt64((meta["seed"] as? Int) ?? 42)
        let desktop = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop")

        let encoder = try await QwenVLPromptEncoder.load(snapshot: Self.modelDir)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: Self.modelDir.appendingPathComponent("vae"), dtype: .float32)

        // apply() mutates the transformer, so each variant gets a freshly loaded one.
        func freshTransformer() throws -> QwenImageTransformer2DModel {
            try QwenImageEditWeights.loadDiTFromPT(
                directory: Self.modelDir.appendingPathComponent("transformer"), dtype: .bfloat16)
        }
        func render(_ tag: String, _ transformer: QwenImageTransformer2DModel,
                    steps: Int, cfg: Float) throws {
            let gen = QwenImageEditGenerator(encoder: encoder, transformer: transformer, vae: vae)
            let t = Date()
            let (pixels, w, h) = try gen.generate(
                image: image, prompt: prompt, negativePrompt: neg,
                steps: steps, trueCFGScale: cfg, seed: seed, progress: { _, _ in })
            let out = desktop.appendingPathComponent("qie-\(tag).png")
            try GenerateDemoTests.writePNG(pixels: pixels, width: w, height: h, to: out)
            print("[turbo] \(tag): \(w)x\(h) in \(String(format: "%.1f", Date().timeIntervalSince(t)))s -> \(out.path)")
        }

        // Base at 4-step CFG 1.0 (no adapter) — the control for the strength sweep.
        try render("base-4step-cfg1", try freshTransformer(), steps: 4, cfg: 1.0)

        // Strength sweep: 1.0 = documented diffusers default (alpha/rank=0.125); higher
        // pushes the adapter. Resolves whether the small default effect is correct or weak.
        for s in [Float(1.0), 4.0, 8.0] {
            let transformer = try freshTransformer()
            try QwenImageEditLoRA.apply(diffusersLoRA: Self.lora, to: transformer, strength: s)
            try render("lightning-4step-cfg1-s\(Int(s))", transformer, steps: 4, cfg: 1.0)
        }
    }
}
