// Phase 1 (foundation) validation: the diffusers -> engine LoRA remapper, exercised
// against the REAL LightX2V Qwen-Image-Edit-2511 Lightning-4step adapter. No DiT
// weights and no 4-step run here — this isolates the key-map / transpose / scale-bake
// pipeline (QwenImageEditLoRA.parameters) so it can be validated cheaply.
//
// Run: QIE_LORA=1 swift test --filter LoRARemapTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class LoRARemapTests: XCTestCase {
    static let loraURL = URL(
        fileURLWithPath:
            "/Users/dustinnielson/Development/telestyle-work/loras/"
            + "QIE-2511-Lightning-4steps-V1.0-bf16.safetensors")

    static let blockKeys: Set<String> = [
        "attn.to_q", "attn.to_k", "attn.to_v",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_add_out", "attn.to_out.0",
        "img_mlp.proj_in", "img_mlp.proj_out",
        "txt_mlp.proj_in", "txt_mlp.proj_out",
    ]

    func testRemap() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_LORA"] == "1", "QIE_LORA=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.loraURL.path),
            "Lightning LoRA not present at \(Self.loraURL.path)")

        let (params, _, targetKeys) = try QwenImageEditLoRA.combined(
            [(Self.loraURL, 1.0)], dtype: .bfloat16)

        // 60 blocks x 12 Linears x {lora_a, lora_b} = 1440 parameter tensors.
        XCTAssertEqual(params.count, 1440, "expected 1440 lora_a/lora_b tensors")
        XCTAssertEqual(targetKeys, Self.blockKeys, "targets must be exactly the 12/block")

        var aCount = 0
        var bCount = 0
        for (key, value) in params {
            // Every key is transformer_blocks.<0-59>.<one of the 12>.lora_{a,b}.
            XCTAssertTrue(key.hasPrefix("transformer_blocks."), key)
            let isA = key.hasSuffix(".lora_a")
            let isB = key.hasSuffix(".lora_b")
            XCTAssertTrue(isA || isB, "unexpected param key \(key)")
            let stripped = String(key.dropFirst("transformer_blocks.".count))
            let parts = stripped.split(separator: ".", maxSplits: 1)
            guard let idx = Int(parts[0]), (0..<60).contains(idx) else {
                return XCTFail("bad block index in \(key)")
            }
            let rel = String(parts[1].dropLast(".lora_a".count))  // strip suffix
            XCTAssertTrue(Self.blockKeys.contains(rel), "unmapped module path \(rel)")

            // Transpose check: lora_a is [in, rank], lora_b is [rank, out], rank == 64.
            if isA { aCount += 1; XCTAssertEqual(value.dim(1), 64, "lora_a rank for \(key)") }
            if isB { bCount += 1; XCTAssertEqual(value.dim(0), 64, "lora_b rank for \(key)") }
        }
        XCTAssertEqual(aCount, 720)
        XCTAssertEqual(bCount, 720)

        // Spot-check a known asymmetric pair: img_mlp.proj_out has in=12288, out=3072,
        // so after transpose lora_a is [12288, 64] and lora_b is [64, 3072].
        let a = params["transformer_blocks.0.img_mlp.proj_out.lora_a"]
        let b = params["transformer_blocks.0.img_mlp.proj_out.lora_b"]
        XCTAssertEqual(a?.shape, [12288, 64])
        XCTAssertEqual(b?.shape, [64, 3072])
    }
}
