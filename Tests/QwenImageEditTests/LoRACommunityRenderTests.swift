// De-risk gate for the generic community-LoRA loader: prove that an arbitrary community
// Qwen-Image-Edit LoRA — including the `diffusion_model.`-prefixed dialect — rank-stacks
// onto the Lightning base via the existing apply() path and renders its trained effect on
// REAL activations. Static parsing is covered by LoRADialectTests; this is the end-to-end
// half (the thing only a real 4-step render can show).
//
// Renders to ~/Desktop for eyeball comparison, control vs. effect (same prompt, Lightning
// on both so 4-step is fair):
//   qie-comm-anime-ctrl   Lightning only          + "transform into anime"
//   qie-comm-anime-lora   Lightning + Photo-to-Anime (diffusion_model. dialect)
//   qie-comm-pixar-ctrl   Lightning only          + "Transform it into Pixar-inspired 3D"
//   qie-comm-pixar-lora   Lightning + Pixar-Inspired-3D (clean transformer. dialect)
//
// Uses the pre-quantized int4 DiT + int4 VL path (~21 GB resident) for speed.
// Run: QIE_COMMUNITY=1 swift test --filter LoRACommunityRenderTests

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import QwenImageEdit

final class LoRACommunityRenderTests: XCTestCase {
    static let modelDir = URL(fileURLWithPath:
        "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")
    static let quantizedDiT = "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-dit-int4-mod8.safetensors"
    static let quantizedEnc = "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-vl7b-int4.safetensors"
    static let loraDir = URL(fileURLWithPath: "/Users/dustinnielson/Development/telestyle-work/loras")
    static let lightning = loraDir.appendingPathComponent("QIE-2511-Lightning-4steps-V1.0-bf16.safetensors")
    static let anime = loraDir.appendingPathComponent("Photo-to-Anime.safetensors")        // diffusion_model. dialect
    static let pixar = loraDir.appendingPathComponent("Pixar-Inspired-3D.safetensors")      // transformer. dialect
    static let input = URL(fileURLWithPath: "/Users/dustinnielson/Desktop/lens-t2i-package.png")

    func testCommunityLoRARender() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIE_COMMUNITY"] == "1", "QIE_COMMUNITY=1")
        for p in [Self.quantizedDiT, Self.quantizedEnc, Self.lightning.path, Self.anime.path,
                  Self.pixar.path, Self.input.path] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }

        let image = try EncoderParityTests.loadRGB(url: Self.input)
        let desktop = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop")

        // Shared, loaded once: int4 VL encoder + bf16 VAE.
        let encoder = try await QwenVLPromptEncoder.load(
            snapshot: Self.modelDir, quantizedTextModelPath: Self.quantizedEnc)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: Self.modelDir.appendingPathComponent("vae"), dtype: .bfloat16)

        // apply() mutates the DiT, so each LoRA combo gets a freshly loaded int4 transformer.
        func generator(_ loras: [(url: URL, strength: Float)]) throws -> QwenImageEditGenerator {
            let dit = try QwenImageEditWeights.loadQuantizedDiT(from: URL(fileURLWithPath: Self.quantizedDiT))
            try QwenImageEditLoRA.apply(diffusersLoRAs: loras, to: dit)
            return QwenImageEditGenerator(encoder: encoder, transformer: dit, vae: vae)
        }
        func render(_ tag: String, _ gen: QwenImageEditGenerator, prompt: String) async throws {
            let t = Date()
            let (px, w, h) = try await gen.generate(
                image: image, prompt: prompt, negativePrompt: " ",
                steps: 4, trueCFGScale: 1.0, seed: 42, progress: { _, _ in })
            let out = desktop.appendingPathComponent("qie-comm-\(tag).png")
            try GenerateDemoTests.writePNG(pixels: px, width: w, height: h, to: out)
            print("[community] \(tag): \(w)x\(h) in \(String(format: "%.1f", Date().timeIntervalSince(t)))s -> \(out.path)")
        }

        let animePrompt = "transform into anime"
        let pixarPrompt = "Transform it into Pixar-inspired 3D"

        // Controls: Lightning only, both prompts (DiT is prompt-independent, reuse it).
        do {
            let ctrl = try generator([(Self.lightning, 4.0)])
            try await render("anime-ctrl", ctrl, prompt: animePrompt)
            try await render("pixar-ctrl", ctrl, prompt: pixarPrompt)
        }
        GPU.clearCache()
        // Photo-to-Anime — exercises the new diffusion_model. prefix dialect on real weights.
        do { try await render("anime-lora", try generator([(Self.lightning, 4.0), (Self.anime, 1.0)]),
                        prompt: animePrompt) }
        GPU.clearCache()
        // Pixar-Inspired-3D — clean transformer. dialect.
        do { try await render("pixar-lora", try generator([(Self.lightning, 4.0), (Self.pixar, 1.0)]),
                        prompt: pixarPrompt) }
    }
}
