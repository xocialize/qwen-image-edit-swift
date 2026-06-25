// In-engine DiT-LoRA for the Qwen-Image-Edit-2511 transformer.
//
// Loads a diffusers-format Qwen-Image LoRA (e.g. LightX2V Qwen-Image-Lightning, the
// 4/8-step DMD distillation adapter) and applies it to `QwenImageTransformer2DModel`
// at runtime, or fuses it permanently into the base Linears for a zero-overhead
// fast-tier snapshot.
//
// Why this exists instead of `MLXLMCommon.LoRAContainer`: that container's
// `load(into:)`/`fuse(with:)` are typed against `LanguageModel`, so a DiT can't call
// them. The underlying primitives (`LoRALinear.from`, `LoRALayer.fused()`, the
// replace-children pattern, `Module.update(parameters:)`) are generic, so we mirror
// the container's ~20-line replace/update logic here, DiT-typed.
//
// LoRA layout (verified against QIE-2511-Lightning-4steps-V1.0-bf16, rank 64):
//   * 12 Linears per block: attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,
//     to_add_out,to_out.0} + {img,txt}_mlp.{proj_in,proj_out}. NOTHING else
//     (img_mod/txt_mod/img_in/txt_in/proj_out/norm_out/time_text_embed are untouched),
//     so application is driven by KEYS PRESENT in the file — never the mlx-lm
//     "all Linear" default, which would over-target.
//   * Diffusers names: attn keys map 1:1 to our @ModuleInfo keys; MLP keys need the
//     same `.net.0.proj.`->`.proj_in.` / `.net.2.`->`.proj_out.` remap as the base
//     weights (see Weights.sanitizeDiTKey).
//   * MLX layout: loraA = diffusers lora_A.T [in,rank]; loraB = lora_B.T [rank,out].
//     Per-layer scale (alpha/rank) is baked into loraB so the LoRALinear scale is 1.0
//     (sidesteps the container's single-global-scale limitation; numerically exact).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `LoRAModel` conformance lets the DiT participate in the mlx-swift-lm LoRA ecosystem.
/// The default `loraDefaultKeys` (all Linear) is deliberately overridden: Qwen-Image
/// LoRAs only adapt attention + the two feed-forwards, never the modulation/in/out
/// projections.
extension QwenImageTransformer2DModel: LoRAModel {
    public var loraLayers: [Module] { transformerBlocks }

    /// Block-relative module paths a Qwen-Image LoRA may target (the 12 per block).
    public var loraDefaultKeys: [String] {
        [
            "attn.to_q", "attn.to_k", "attn.to_v",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
            "attn.to_add_out", "attn.to_out.0",
            "img_mlp.proj_in", "img_mlp.proj_out",
            "txt_mlp.proj_in", "txt_mlp.proj_out",
        ]
    }
}

public enum QwenImageEditLoRA {

    enum LoRAError: Error, LocalizedError {
        case incompleteTriple(String)
        case noTargets(String)

        var errorDescription: String? {
            switch self {
            case .incompleteTriple(let p):
                return "LoRA layer \(p) is missing a lora_A / lora_B / alpha tensor."
            case .noTargets(let url):
                return "No recognizable Qwen-Image LoRA tensors found in \(url)."
            }
        }
    }

    private static let loraASuffix = ".lora_A.default.weight"
    private static let loraBSuffix = ".lora_B.default.weight"
    private static let alphaSuffix = ".alpha"

    /// diffusers MLP child names -> our `QwenFeedForward` @ModuleInfo keys. Attention
    /// names already match, so this is the only rename needed (mirrors sanitizeDiTKey).
    private static func remap(_ basePath: String) -> String {
        basePath
            .replacingOccurrences(of: ".net.0.proj", with: ".proj_in")
            .replacingOccurrences(of: ".net.2", with: ".proj_out")
    }

    /// Strip the `transformer_blocks.<i>.` prefix to get the block-relative module path.
    private static func blockRelative(_ modelPath: String) -> String? {
        let parts = modelPath.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "transformer_blocks" else { return nil }
        return String(parts[2])
    }

    /// Parse a diffusers LoRA into engine-keyed `lora_a` / `lora_b` parameters (scale
    /// baked into `lora_b`) plus the set of block-relative target module paths.
    static func parameters(
        from url: URL, dtype: DType
    ) throws -> (params: [String: MLXArray], targetKeys: Set<String>) {
        let raw = try MLX.loadArrays(url: url)

        // Group raw tensors by remapped, model-root-relative base path.
        var a: [String: MLXArray] = [:]
        var b: [String: MLXArray] = [:]
        var alpha: [String: MLXArray] = [:]
        for (key, value) in raw {
            if key.hasSuffix(loraASuffix) {
                a[remap(String(key.dropLast(loraASuffix.count)))] = value
            } else if key.hasSuffix(loraBSuffix) {
                b[remap(String(key.dropLast(loraBSuffix.count)))] = value
            } else if key.hasSuffix(alphaSuffix) {
                alpha[remap(String(key.dropLast(alphaSuffix.count)))] = value
            }
        }
        guard !a.isEmpty else { throw LoRAError.noTargets(url.path) }

        var params: [String: MLXArray] = [:]
        var targetKeys: Set<String> = []
        for (base, aMat) in a {
            guard let bMat = b[base], let alphaScalar = alpha[base] else {
                throw LoRAError.incompleteTriple(base)
            }
            let rank = aMat.dim(0)  // diffusers lora_A is [rank, in]
            let scale = alphaScalar.item(Float.self) / Float(rank)
            // MLX LoRALinear: y + (x @ loraA[in,rank]) @ loraB[rank,out]; bake scale in B.
            params[base + ".lora_a"] = aMat.T.asType(dtype)
            params[base + ".lora_b"] = (scale * bMat.T).asType(dtype)
            if let rel = blockRelative(base) { targetKeys.insert(rel) }
        }
        return (params, targetKeys)
    }

    /// Apply a diffusers-format Qwen-Image LoRA to the DiT in place (runtime adapter).
    /// Recommended for step-distillation LoRAs: the low-rank term is added in the
    /// activation path, so it survives bf16 (unlike `fuse` into a bf16 weight). Overhead
    /// is one extra rank-r matmul per adapted Linear — negligible against the 20B base.
    public static func apply(
        diffusersLoRA url: URL,
        to model: QwenImageTransformer2DModel,
        dtype: DType = .bfloat16
    ) throws {
        let (params, targetKeys) = try parameters(from: url, dtype: dtype)
        replaceTargets(in: model, keys: targetKeys) { linear in
            LoRALinear.from(linear: linear, rank: 64, scale: 1.0)  // scale baked into lora_b
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
    }

    /// Apply, then permanently fuse the adapter into the base Linears (W += B·A).
    ///
    /// PRECISION CAVEAT: fusing folds the low-rank delta into the stored weight, so the
    /// delta must be representable at the weight's ULP. Step-distillation LoRAs (e.g.
    /// Qwen-Image-Lightning) have per-weight deltas (~1e-4) BELOW bf16's ULP at the base
    /// scale — fusing into bf16 (or fp16) silently rounds them away (see
    /// LoRAFuseTests.testBf16FusionIsLossy). For those, use `apply` (the low-rank term is
    /// added in the activation path and survives bf16). `fuse` is for fp32 storage or
    /// LoRAs whose deltas exceed the storage ULP. The arithmetic itself is exact in fp32
    /// (LoRAFuseTests.testFuseArithmeticFP32).
    public static func fuse(
        diffusersLoRA url: URL,
        into model: QwenImageTransformer2DModel,
        dtype: DType = .bfloat16
    ) throws {
        let (params, targetKeys) = try parameters(from: url, dtype: dtype)
        replaceTargets(in: model, keys: targetKeys) { linear in
            LoRALinear.from(linear: linear, rank: 64, scale: 1.0)
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
        // Collapse every LoRALinear back into a plain fused Linear.
        for block in model.transformerBlocks {
            var fused: [(String, Module)] = []
            for (key, child) in block.namedModules() where targetKeys.contains(key) {
                if let lora = child as? LoRALayer {
                    fused.append((key, lora.fused()))
                }
            }
            if !fused.isEmpty { block.update(modules: .unflattened(fused)) }
        }
    }

    /// Replace each targeted leaf `Linear` in every block with a transformed module.
    private static func replaceTargets(
        in model: QwenImageTransformer2DModel,
        keys: Set<String>,
        _ transform: (Linear) -> Module
    ) {
        for block in model.transformerBlocks {
            var update: [(String, Module)] = []
            for (key, child) in block.namedModules() where keys.contains(key) {
                if let linear = child as? Linear {
                    update.append((key, transform(linear)))
                }
            }
            if !update.isEmpty { block.update(modules: .unflattened(update)) }
        }
    }
}
