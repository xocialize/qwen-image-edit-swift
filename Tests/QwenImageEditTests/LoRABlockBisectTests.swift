// Block-level bisect for the bf16 + LoRA corruption (AGENT_BRIDGE 2026-07-25).
//
// Observed at the pipeline level: bf16 + LoRA renders banded static above ~1366 image
// tokens; int8 + LoRA is clean; bf16 WITHOUT LoRA is clean at the same M. Eliminated so
// far: the FFN row-chunk (corrupt with it on and off), the adapter GEMM shapes in
// isolation (all cos 0.9999988), non-square, bf16 alone.
//
// This probe runs ONE real-size QwenTransformerBlock (dim 3072, heads 24×128 — the actual
// model geometry) with random weights at M ∈ {1024, 4096}, bf16 vs an fp32 twin holding
// bit-identical (cast) weights, with synthetic rank-32 LoRA factors attached via the SAME
// machinery the swapper uses (LoRALinear.from + update(modules:) + update(parameters:)).
// Key families toggle independently, so one run localizes the corruption to attn / mlp /
// mod — or exonerates the block entirely and pushes the search up to the pipeline.
//
// No 20B load; seconds per case. GPU stream (the failure is GPU-kernel-shaped).
//
// Run: QIE_LORA_BISECT=1 swift test --filter LoRABlockBisectTests

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import QwenImageEdit

final class LoRABlockBisectTests: XCTestCase {

    static let dim = 3072
    static let heads = 24
    static let headDim = 128
    static let txtLen = 27

    /// Deterministic filler, same values for both dtype twins.
    private static func fill(_ shape: [Int], seed: UInt64, scale: Float = 0.02) -> MLXArray {
        var lcg = seed &* 6_364_136_223_846_793_005 &+ 1

        let n = shape.reduce(1, *)
        var v = [Float](repeating: 0, count: n)
        for i in 0..<n {
            lcg = lcg &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            v[i] = (Float(Int64(bitPattern: lcg >> 11)) / Float(Int64.max >> 11)) * scale
        }
        return MLXArray(v, shape)
    }

    /// Base weights for one block, keyed by the block's parameter paths (fp32 master).
    private static func blockWeights() -> [String: MLXArray] {
        var w: [String: MLXArray] = [:]
        var seed: UInt64 = 7
        func add(_ key: String, _ shape: [Int]) {
            seed &+= 1
            w[key] = fill(shape, seed: seed)
        }
        let d = dim, inner = heads * headDim
        for stream in ["", "add_"] {
            add("attn.to_\(stream.isEmpty ? "q" : "")\(stream.isEmpty ? "" : "q_proj").weight", [inner, d])
        }
        // simpler: enumerate explicitly
        w.removeAll()
        seed = 7
        for k in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
                  "attn.add_v_proj", "attn.to_add_out"] {
            add("\(k).weight", [inner, d]); add("\(k).bias", [inner == d ? d : inner])
        }
        add("attn.to_out.0.weight", [d, inner]); add("attn.to_out.0.bias", [d])
        for k in ["attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k"] {
            add("\(k).weight", [headDim])
        }
        for s in ["img", "txt"] {
            add("\(s)_mod.weight", [6 * d, d]); add("\(s)_mod.bias", [6 * d])
            add("\(s)_mlp.proj_in.weight", [4 * d, d]); add("\(s)_mlp.proj_in.bias", [4 * d])
            add("\(s)_mlp.proj_out.weight", [d, 4 * d]); add("\(s)_mlp.proj_out.bias", [d])
        }
        return w
    }

    private static func makeBlock(dtype: DType, weights: [String: MLXArray]) throws
        -> QwenTransformerBlock
    {
        let block = QwenTransformerBlock(
            dim: dim, numAttentionHeads: heads, attentionHeadDim: headDim)
        let cast = weights.mapValues { $0.asType(dtype) }
        try block.update(parameters: ModuleParameters.unflattened(cast), verify: .noUnusedKeys)
        eval(block)
        return block
    }

    /// Attach synthetic rank-32 LoRA factors to `keys` on a standalone block, using the same
    /// LoRALinear.from + update(modules:) + update(parameters:) path the swapper uses.
    private static func attachLoRA(
        _ block: QwenTransformerBlock, keys: [String], dtype: DType, rank: Int = 32
    ) throws {
        var moduleUpdate: [(String, Module)] = []
        var params: [String: MLXArray] = [:]
        var seed: UInt64 = 999
        for (key, child) in block.namedModules() where keys.contains(key) {
            guard let linear = child as? Linear else { continue }
            moduleUpdate.append((key, LoRALinear.from(linear: linear, rank: rank, scale: 1.0)))
            let (out, inp) = linear.shape
            seed &+= 2
            // Non-trivial factors so the low-rank term contributes real signal.
            params["\(key).lora_a"] = fill([inp, rank], seed: seed, scale: 0.05).asType(dtype)
            params["\(key).lora_b"] = fill([rank, out], seed: seed &+ 1, scale: 0.05).asType(dtype)
        }
        XCTAssertFalse(moduleUpdate.isEmpty, "no LoRA targets matched \(keys)")
        block.update(modules: ModuleChildren.unflattened(moduleUpdate))
        try block.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
        eval(block)
    }

    private static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).flattened()
        let y = b.asType(.float32).flattened()
        let c = sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)) + 1e-12)
        eval(c)
        return c.item(Float.self)
    }

    private func runBlock(
        _ block: QwenTransformerBlock, dtype: DType, m: Int
    ) -> (img: MLXArray, txt: MLXArray) {
        let hidden = Self.fill([1, m, Self.dim], seed: 42, scale: 0.5).asType(dtype)
        let encoder = Self.fill([1, Self.txtLen, Self.dim], seed: 43, scale: 0.5).asType(dtype)
        let temb = Self.fill([2, Self.dim], seed: 44, scale: 0.5).asType(dtype)
        let imgRope = (
            cos(Self.fill([m, Self.headDim / 2], seed: 45, scale: 3)).asType(dtype),
            sin(Self.fill([m, Self.headDim / 2], seed: 45, scale: 3)).asType(dtype))
        let txtRope = (
            cos(Self.fill([Self.txtLen, Self.headDim / 2], seed: 46, scale: 3)).asType(dtype),
            sin(Self.fill([Self.txtLen, Self.headDim / 2], seed: 46, scale: 3)).asType(dtype))
        let (txt, img) = block(
            hiddenStates: hidden, encoderHiddenStates: encoder, temb: temb,
            imageRotaryEmb: (imgRope, txtRope), attentionMask: nil, modulateIndex: nil)
        eval(img, txt)
        return (img, txt)
    }

    /// Same twin-block comparison, but with the REAL adapter's block-0 factors (dialect
    /// remap, trained magnitudes, alpha/rank scaling baked into B) instead of synthetic
    /// fill. Discriminates "the machinery is fine, the numbers break it" from "the
    /// machinery is fine, full stop".
    func testRealFactorsBlock0() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_LORA_BISECT"] == "1", "QIE_LORA_BISECT=1")
        let url = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QIE_LORA_FILE"]
            ?? "/Volumes/DEV_ARCHIVE/models/xocialize/qwen-image-flash-loras/loras/day-of-the-tentacle.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "missing \(url.path)")

        // Master factors in fp32; each twin casts to its own dtype (bf16 twin = production).
        let master = try QwenImageEditLoRA.factors(from: url, dtype: .float32, strength: 1.0)
        let prefix = "transformer_blocks.0."
        var rel: [String: (a: MLXArray, b: MLXArray)] = [:]
        for (base, f) in master where base.hasPrefix(prefix) {
            rel[String(base.dropFirst(prefix.count))] = (f.a, f.b)
        }
        XCTAssertFalse(rel.isEmpty, "no block-0 factors in \(url.lastPathComponent)")
        print("[bisect-real] block-0 targets: \(rel.keys.sorted())")
        for (k, f) in rel {
            let aMax = abs(f.a).max().item(Float.self)
            let bMax = abs(f.b).max().item(Float.self)
            print(String(format: "[bisect-real]   %@ rank=%d |a|max %.4f |b|max %.4f",
                         k as NSString, f.a.dim(1), aMax, bMax))
        }

        func attach(_ block: QwenTransformerBlock, dtype: DType) throws {
            var moduleUpdate: [(String, Module)] = []
            var params: [String: MLXArray] = [:]
            for (key, child) in block.namedModules() {
                guard let f = rel[key], let linear = child as? Linear else { continue }
                moduleUpdate.append(
                    (key, LoRALinear.from(linear: linear, rank: f.a.dim(1), scale: 1.0)))
                params["\(key).lora_a"] = f.a.asType(dtype)
                params["\(key).lora_b"] = f.b.asType(dtype)
            }
            block.update(modules: ModuleChildren.unflattened(moduleUpdate))
            try block.update(
                parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
            eval(block)
        }

        let weights = Self.blockWeights()
        for m in [1024, 4096] {
            let b16 = try Self.makeBlock(dtype: .bfloat16, weights: weights)
            let f32 = try Self.makeBlock(dtype: .float32, weights: weights)
            try attach(b16, dtype: .bfloat16)
            try attach(f32, dtype: .float32)
            let outB = runBlock(b16, dtype: .bfloat16, m: m)
            let outF = runBlock(f32, dtype: .float32, m: m)
            let cosImg = Self.cosine(outB.img, outF.img)
            let cosTxt = Self.cosine(outB.txt, outF.txt)
            print(String(
                format: "[bisect-real] M=%d img cos %.6f  txt cos %.6f%@",
                m, cosImg, cosTxt, (cosImg < 0.99 ? "  <-- DIVERGES" : "") as NSString))
        }
    }

    func testBisectKeyFamilies() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_LORA_BISECT"] == "1", "QIE_LORA_BISECT=1")

        let attn = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
                    "attn.add_v_proj", "attn.to_out.0", "attn.to_add_out"]
        let mlp = ["img_mlp.proj_in", "img_mlp.proj_out", "txt_mlp.proj_in", "txt_mlp.proj_out"]
        let mod = ["img_mod", "txt_mod"]

        let families: [(String, [String])] = [
            ("none", []), ("attn", attn), ("mlp", mlp), ("mod", mod),
            ("all", attn + mlp + mod),
        ]

        let master = Self.blockWeights()
        for m in [1024, 4096] {
            print("[bisect] ---- M=\(m) ----")
            for (label, keys) in families {
                // Fresh twins per case: swapping modules mutates the block.
                let b16 = try Self.makeBlock(dtype: .bfloat16, weights: master)
                let f32 = try Self.makeBlock(dtype: .float32, weights: master)
                if !keys.isEmpty {
                    try Self.attachLoRA(b16, keys: keys, dtype: .bfloat16)
                    try Self.attachLoRA(f32, keys: keys, dtype: .float32)
                }
                let outB = runBlock(b16, dtype: .bfloat16, m: m)
                let outF = runBlock(f32, dtype: .float32, m: m)
                let cosImg = Self.cosine(outB.img, outF.img)
                let cosTxt = Self.cosine(outB.txt, outF.txt)
                let flag = cosImg < 0.99 || cosTxt < 0.99 ? "  <-- DIVERGES" : ""
                print(String(
                    format: "[bisect] M=%d %-5@ img cos %.6f  txt cos %.6f%@",
                    m, label as NSString, cosImg, cosTxt, flag as NSString))
            }
        }
    }
}
