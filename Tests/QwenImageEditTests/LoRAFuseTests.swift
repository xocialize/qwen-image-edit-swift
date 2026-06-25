// Phase 1 (foundation) numeric gates for the in-engine DiT-LoRA path, against an
// independent PyTorch reference (diffusers fuse_lora(lora_scale=1.0) convention =
// alpha/rank scaling — what the production TeleStyle merge uses).
// Reference: /tmp/lora_fuse_delta_ref.safetensors via /tmp/gen_lora_fuse_ref.py.
//
//   testFuseArithmeticFP32 — the real correctness gate. No DiT weights: reconstruct
//     each layer's delta through the actual LoRALinear.fused() path in fp32 and match
//     the PyTorch reference. Proves the remap + transpose + scale-bake + fuse formula.
//     Run: QIE_LORA=1 swift test --filter testFuseArithmeticFP32
//
//   testBf16FusionIsLossy — characterization guard. Loads the real base DiT, fuses in
//     bf16, and asserts the stored delta is LARGELY LOST: this Lightning LoRA's
//     per-weight deltas (max ~1.8e-4) sit at/below bf16's ULP (~2e-4) at the base-weight
//     scale (~0.05), so `base + delta` rounds back to `base`. The takeaway encoded here:
//     a fused bf16 (or fp16) snapshot CANNOT carry this LoRA — the fast tier must use the
//     runtime apply() path (low-rank term added in the activation space). fuse() is only
//     valid into fp32 storage or for LoRAs whose deltas exceed the storage ULP.
//     Run: QIE_LORA_FUSE=1 swift test --filter testBf16FusionIsLossy

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import QwenImageEdit

final class LoRAFuseTests: XCTestCase {
    static let baseDiT = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/"
            + "Qwen-Image-Edit-2511/transformer")
    static let loraURL = URL(
        fileURLWithPath:
            "/Users/dustinnielson/Development/telestyle-work/loras/"
            + "QIE-2511-Lightning-4steps-V1.0-bf16.safetensors")
    static let deltaRef = URL(fileURLWithPath: "/tmp/lora_fuse_delta_ref.safetensors")

    private static let sampled = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.img_mlp.proj_out",
        "transformer_blocks.59.attn.to_out.0",
    ]

    private func relFro(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = (a - b).asType(.float32)
        let ref = b.asType(.float32)
        return sqrt((diff * diff).sum()).item(Float.self)
            / sqrt((ref * ref).sum()).item(Float.self)
    }

    func testFuseArithmeticFP32() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_LORA"] == "1", "QIE_LORA=1")
        for url in [Self.loraURL, Self.deltaRef] {
            try XCTSkipUnless(
                FileManager.default.fileExists(atPath: url.path), "missing \(url.path)")
        }
        let (params, _) = try QwenImageEditLoRA.parameters(from: Self.loraURL, dtype: .float32)
        let ref = try MLX.loadArrays(url: Self.deltaRef)

        for name in Self.sampled {
            let a = params[name + ".lora_a"]!  // [in, rank]
            let b = params[name + ".lora_b"]!  // [rank, out]  (alpha/rank baked in)
            let (inDim, rank, outDim) = (a.dim(0), a.dim(1), b.dim(1))

            // Drive the delta through the REAL LoRALinear.fused() path on a zero base, so
            // fused().weight == 0 + delta. Exercises mlx-swift-lm's fuse math end to end.
            let zero = Linear(weight: MLXArray.zeros([outDim, inDim], dtype: .float32))
            guard let lora = LoRALinear.from(linear: zero, rank: rank, scale: 1.0) as? LoRALinear
            else { return XCTFail("LoRALinear.from did not yield a LoRALinear") }
            try lora.update(
                parameters: ModuleParameters.unflattened(["lora_a": a, "lora_b": b]),
                verify: .noUnusedKeys)
            guard let fused = lora.fused() as? Linear else {
                return XCTFail("fused() did not yield a Linear")
            }
            let rel = relFro(fused.weight, ref[name]!)
            print("[fp32] \(name): rel-fro \(rel)")
            XCTAssertLessThan(rel, 1e-3, "\(name) fp32 fuse must match PyTorch reference")
        }
    }

    func testBf16FusionIsLossy() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_LORA_FUSE"] == "1", "QIE_LORA_FUSE=1")
        for url in [Self.baseDiT, Self.loraURL, Self.deltaRef] {
            try XCTSkipUnless(
                FileManager.default.fileExists(atPath: url.path), "missing \(url.path)")
        }
        let model = try QwenImageEditWeights.loadDiTFromPT(
            directory: Self.baseDiT, dtype: .bfloat16)

        let base0 = model.transformerBlocks[0].attn.toQ.weight
        let base1 = model.transformerBlocks[0].imgMLP.projOut.weight
        let base2 = model.transformerBlocks[59].attn.toOut[0].weight
        eval(base0, base1, base2)

        try QwenImageEditLoRA.fuse(diffusersLoRA: Self.loraURL, into: model)  // throws on bad keys

        let ref = try MLX.loadArrays(url: Self.deltaRef)
        let deltas: [(String, MLXArray)] = [
            (Self.sampled[0], model.transformerBlocks[0].attn.toQ.weight - base0),
            (Self.sampled[1], model.transformerBlocks[0].imgMLP.projOut.weight - base1),
            (Self.sampled[2], model.transformerBlocks[59].attn.toOut[0].weight - base2),
        ]
        for (name, deltaEng) in deltas {
            let rel = relFro(deltaEng, ref[name]!)
            print("[bf16] \(name): rel-fro vs fp32 ref \(rel)  (>0.8 => delta lost in bf16)")
            // Most of the delta is below bf16 ULP and rounds away — guard that this stays
            // true so nobody ships a fused-bf16 fast tier expecting the LoRA to be present.
            XCTAssertGreaterThan(rel, 0.8, "bf16 fusion unexpectedly preserved the delta")
        }
    }
}
