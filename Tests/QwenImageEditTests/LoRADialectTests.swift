// Validates the LoRA key-dialect handling added for the generic community-LoRA loader.
// Pure string-mapping — no weights, no GPU — so it always runs (no env gate).
//
// Dialects (header-probed from the prithivMLmods Qwen-Image-Edit-2511 LoRA collection):
//   * diffusers PEFT  `transformer.…lora_A.weight`               (already supported)
//   * diffusers alt   `transformer.…lora.down.weight`            (Anything2Real)
//   * diffusion_model `diffusion_model.…lora_A.weight`           (Photo-to-Anime, Upscaler, …)
//   * kohya           `lora_unet_…_attn_add_k_proj.lora_down.weight` (Manga-Tone)

import XCTest

@testable import QwenImageEdit

final class LoRADialectTests: XCTestCase {
    private let known: Set<String> = [
        "attn.to_q", "attn.to_k", "attn.to_v",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_add_out", "attn.to_out.0",
        "img_mlp.proj_in", "img_mlp.proj_out",
        "txt_mlp.proj_in", "txt_mlp.proj_out",
        "img_mod", "txt_mod",
    ]

    /// Each base (the key with its lora_A/B/down/up suffix already stripped) must remap to a
    /// `transformer_blocks.<n>.<rel>` path whose `rel` is one of the engine's adapted modules.
    private func assertMapsTo(_ base: String, block: Int, rel: String, line: UInt = #line) {
        let mapped = QwenImageEditLoRA.remap(base)
        XCTAssertEqual(mapped, "transformer_blocks.\(block).\(rel)", base, line: line)
        XCTAssertEqual(QwenImageEditLoRA.blockRelative(mapped), rel, base, line: line)
        XCTAssertTrue(known.contains(rel), "rel \(rel) not an adapted module", line: line)
    }

    func testDiffusersPEFTPrefix() {
        assertMapsTo("transformer.transformer_blocks.0.attn.to_q", block: 0, rel: "attn.to_q")
        assertMapsTo("transformer.transformer_blocks.59.attn.to_out.0", block: 59, rel: "attn.to_out.0")
    }

    func testDiffusionModelPrefix() {  // Photo-to-Anime, Upscaler, Style-Transfer, Any-light
        assertMapsTo("diffusion_model.transformer_blocks.0.attn.add_k_proj", block: 0, rel: "attn.add_k_proj")
        assertMapsTo("diffusion_model.transformer_blocks.7.attn.to_v", block: 7, rel: "attn.to_v")
    }

    func testMLPAndModNormalization() {
        assertMapsTo("transformer.transformer_blocks.3.img_mlp.net.0.proj", block: 3, rel: "img_mlp.proj_in")
        assertMapsTo("transformer.transformer_blocks.3.txt_mlp.net.2", block: 3, rel: "txt_mlp.proj_out")
        assertMapsTo("transformer.transformer_blocks.3.img_mod.1", block: 3, rel: "img_mod")
        assertMapsTo("diffusion_model.transformer_blocks.3.txt_mod.1", block: 3, rel: "txt_mod")
    }

    func testKohyaUnderscoreFlattening() {  // Manga-Tone
        assertMapsTo("lora_unet_transformer_blocks_0_attn_add_k_proj", block: 0, rel: "attn.add_k_proj")
        assertMapsTo("lora_unet_transformer_blocks_12_attn_to_out_0", block: 12, rel: "attn.to_out.0")
        assertMapsTo("lora_unet_transformer_blocks_5_img_mlp_net_0_proj", block: 5, rel: "img_mlp.proj_in")
        assertMapsTo("lora_unet_transformer_blocks_5_txt_mlp_net_2", block: 5, rel: "txt_mlp.proj_out")
        assertMapsTo("lora_unet_transformer_blocks_8_img_mod_1", block: 8, rel: "img_mod")
    }

    func testDekohyaRejectsNonKohya() {
        XCTAssertNil(QwenImageEditLoRA.dekohya("transformer.transformer_blocks.0.attn.to_q"))
        XCTAssertNil(QwenImageEditLoRA.dekohya("lora_unet_transformer_blocks_0_unknown_module"))
    }
}
