// In-engine DiT-LoRA for the Qwen-Image-Edit-2511 transformer.
//
// Loads one or more diffusers-format Qwen-Image LoRAs (e.g. LightX2V Qwen-Image-
// Lightning, the 4/8-step DMD distillation adapter; the TeleStyleV2 style adapter) and
// applies them to `QwenImageTransformer2DModel` at runtime, or fuses a single one into
// the base Linears.
//
// Why this exists instead of `MLXLMCommon.LoRAContainer`: that container's
// `load(into:)`/`fuse(with:)` are typed against `LanguageModel`, so a DiT can't call
// them. The underlying primitives (`LoRALinear.from`, `LoRALayer.fused()`, the
// replace-children pattern, `Module.update(parameters:)`) are generic, so we mirror
// the container's ~20-line replace/update logic here, DiT-typed.
//
// Two LoRA dialects are handled (verified against the QIE-2511 Lightning + TeleStyleV2
// adapters):
//   * keys: optional `transformer.` prefix; suffix `.lora_{A,B}.weight` OR
//     `.lora_{A,B}.default.weight`; optional per-layer `.alpha`.
//   * module remap to engine @ModuleInfo keys (mirrors Weights.sanitizeDiTKey):
//     `img_mod.1`/`txt_mod.1` -> `img_mod`/`txt_mod`; `net.0.proj`/`net.2` ->
//     `proj_in`/`proj_out`; attention names already match.
//   * scale: `strength * alpha/rank` when alpha is present, else `strength` (alpha-less
//     adapters apply at scale 1.0 — confirmed against the fused snapshot). Baked into
//     `lora_b` so the LoRALinear scale is 1.0.
//   * application is driven by KEYS PRESENT in each file (never the mlx-lm "all Linear"
//     default, which would over-target). Multiple LoRAs combine by rank-stacking, so the
//     adapted layer's contribution is the exact SUM of each adapter's low-rank term.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `LoRAModel` conformance lets the DiT participate in the mlx-swift-lm LoRA ecosystem.
extension QwenImageTransformer2DModel: LoRAModel {
    public var loraLayers: [Module] { transformerBlocks }

    /// The block-relative Linears Qwen-Image LoRAs are known to adapt (Lightning targets
    /// attn + both MLP projections; the TeleStyleV2 style adapter additionally targets
    /// the modulation linears). Illustrative only — actual application is keys-present per
    /// file, so this deliberately omits the all-Linear default that would over-target.
    public var loraDefaultKeys: [String] {
        [
            "attn.to_q", "attn.to_k", "attn.to_v",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
            "attn.to_add_out", "attn.to_out.0",
            "img_mlp.proj_in", "img_mlp.proj_out",
            "txt_mlp.proj_in", "txt_mlp.proj_out",
            "img_mod", "txt_mod",
        ]
    }
}

public enum QwenImageEditLoRA {

    enum LoRAError: Error, LocalizedError {
        case incompletePair(String)
        case noTargets(String)

        var errorDescription: String? {
            switch self {
            case .incompletePair(let p):
                return "LoRA layer \(p) is missing its lora_A or lora_B tensor."
            case .noTargets(let url):
                return "No recognizable Qwen-Image LoRA tensors found in \(url)."
            }
        }
    }

    // A/B factor suffixes across the LoRA dialects in the wild (verified by header-probing
    // the prithivMLmods/dx8152/fal/autoweeb/kohya community Qwen-Image-Edit adapters):
    //   * diffusers PEFT:  `.lora_A.weight` / `.lora_A.default.weight` (and _B)
    //   * diffusers alt:   `.lora.down.weight` / `.lora.up.weight`   (e.g. Anything2Real)
    //   * kohya/sd-scripts:`.lora_down.weight` / `.lora_up.weight`   (e.g. Manga-Tone)
    // down == A (the [rank, in] projection); up == B (the [out, rank]).
    private static let aSuffixes = [
        ".lora_A.default.weight", ".lora_A.weight", ".lora.down.weight", ".lora_down.weight",
    ]
    private static let bSuffixes = [
        ".lora_B.default.weight", ".lora_B.weight", ".lora.up.weight", ".lora_up.weight",
    ]
    private static let alphaSuffix = ".alpha"

    /// kohya flattens the module path to underscores under a `lora_unet_` prefix (and the
    /// submodule names themselves contain underscores, so a naive `_`->`.` is wrong). Map the
    /// known block-relative submodules explicitly back to their dotted form (pre-remap, so the
    /// `.net.0.proj`/`.img_mod.1` normalizations below still fire).
    private static let kohyaSubmodule: [String: String] = [
        "attn_to_q": "attn.to_q", "attn_to_k": "attn.to_k", "attn_to_v": "attn.to_v",
        "attn_add_q_proj": "attn.add_q_proj", "attn_add_k_proj": "attn.add_k_proj",
        "attn_add_v_proj": "attn.add_v_proj",
        "attn_to_add_out": "attn.to_add_out", "attn_to_out_0": "attn.to_out.0",
        "img_mlp_net_0_proj": "img_mlp.net.0.proj", "img_mlp_net_2": "img_mlp.net.2",
        "txt_mlp_net_0_proj": "txt_mlp.net.0.proj", "txt_mlp_net_2": "txt_mlp.net.2",
        "img_mod_1": "img_mod.1", "txt_mod_1": "txt_mod.1",
    ]

    /// Convert a kohya `lora_unet_transformer_blocks_<n>_<submodule>` base to the dotted
    /// `transformer_blocks.<n>.<submodule>` form. Returns nil for non-kohya / unknown paths.
    static func dekohya(_ s: String) -> String? {
        guard s.hasPrefix("lora_unet_") else { return nil }
        let body = String(s.dropFirst("lora_unet_".count))
        guard body.hasPrefix("transformer_blocks_") else { return nil }
        let afterBlocks = body.dropFirst("transformer_blocks_".count)
        guard let usc = afterBlocks.firstIndex(of: "_") else { return nil }
        let idx = String(afterBlocks[..<usc])
        guard Int(idx) != nil else { return nil }
        let rest = String(afterBlocks[afterBlocks.index(after: usc)...])
        guard let dotted = kohyaSubmodule[rest] else { return nil }
        return "transformer_blocks.\(idx).\(dotted)"
    }

    /// diffusers/kohya LoRA child path -> engine DiT module path. Handles the `transformer.`
    /// and `diffusion_model.` top-level prefixes and the kohya underscore flattening, then the
    /// shared submodule normalizations (mirrors Weights.sanitizeDiTKey). Attention names match.
    static func remap(_ path: String) -> String {
        var s = dekohya(path) ?? path
        if s.hasPrefix("transformer.") { s.removeFirst("transformer.".count) }
        else if s.hasPrefix("diffusion_model.") { s.removeFirst("diffusion_model.".count) }
        return s
            .replacingOccurrences(of: ".img_mod.1", with: ".img_mod")
            .replacingOccurrences(of: ".txt_mod.1", with: ".txt_mod")
            .replacingOccurrences(of: ".net.0.proj", with: ".proj_in")
            .replacingOccurrences(of: ".net.2", with: ".proj_out")
    }

    /// Strip the `transformer_blocks.<i>.` prefix to get the block-relative module path.
    static func blockRelative(_ modelPath: String) -> String? {
        let parts = modelPath.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "transformer_blocks" else { return nil }
        return String(parts[2])
    }

    /// One LoRA's per-target low-rank factors, keyed by engine module path. `a` is
    /// [in, rank]; `b` is [rank, out] with the layer's effective scale baked in.
    struct Factors { var a: MLXArray; var b: MLXArray }

    static func factors(
        from url: URL, dtype: DType, strength: Float
    ) throws -> [String: Factors] {
        let raw = try MLX.loadArrays(url: url)
        func match(_ key: String, _ suffixes: [String]) -> String? {
            for s in suffixes where key.hasSuffix(s) {
                return remap(String(key.dropLast(s.count)))
            }
            return nil
        }
        var aMats: [String: MLXArray] = [:]
        var bMats: [String: MLXArray] = [:]
        var alphas: [String: MLXArray] = [:]
        for (key, value) in raw {
            if let base = match(key, aSuffixes) { aMats[base] = value }
            else if let base = match(key, bSuffixes) { bMats[base] = value }
            else if key.hasSuffix(alphaSuffix) {
                alphas[remap(String(key.dropLast(alphaSuffix.count)))] = value
            }
        }
        guard !aMats.isEmpty else { throw LoRAError.noTargets(url.path) }

        var out: [String: Factors] = [:]
        for (base, aMat) in aMats {
            guard let bMat = bMats[base] else { throw LoRAError.incompletePair(base) }
            let rank = aMat.dim(0)  // diffusers lora_A is [rank, in]
            // alpha/rank when present; alpha-less adapters apply at scale 1.0.
            let scale = strength * (alphas[base].map { $0.item(Float.self) / Float(rank) } ?? 1.0)
            out[base] = Factors(a: aMat.T.asType(dtype), b: (scale * bMat.T).asType(dtype))
        }
        return out
    }

    /// Combine one or more LoRAs into per-module `lora_a`/`lora_b` parameters by
    /// rank-stacking: concat the `a` factors along rank and the `b` factors along rank, so
    /// the LoRALinear contribution becomes the exact SUM of each adapter's low-rank term —
    /// what diffusers' multi-adapter `fuse_lora` produces. Returns the combined params, the
    /// per-module combined rank, and the block-relative target keys.
    static func combined(
        _ loras: [(url: URL, strength: Float)], dtype: DType
    ) throws -> (params: [String: MLXArray], ranks: [String: Int], targetKeys: Set<String>) {
        let perLoRA = try loras.map { try factors(from: $0.url, dtype: dtype, strength: $0.strength) }
        var bases = Set<String>()
        perLoRA.forEach { bases.formUnion($0.keys) }

        var params: [String: MLXArray] = [:]
        var ranks: [String: Int] = [:]
        var targetKeys = Set<String>()
        for base in bases {
            // Only adapters on a transformer block get a LoRALinear; skip any top-level
            // target (e.g. img_in / time_text_embed) rather than emit an orphan param that
            // would fail update(verify: .noUnusedKeys). Generic loader: skip, don't crash.
            guard let rel = blockRelative(base) else { continue }
            let present = perLoRA.compactMap { $0[base] }  // same order for a and b
            let aCat = present.count == 1 ? present[0].a : concatenated(present.map(\.a), axis: 1)
            let bCat = present.count == 1 ? present[0].b : concatenated(present.map(\.b), axis: 0)
            params[base + ".lora_a"] = aCat
            params[base + ".lora_b"] = bCat
            ranks[base] = aCat.dim(1)
            targetKeys.insert(rel)
        }
        return (params, ranks, targetKeys)
    }

    /// Apply one or more diffusers-format Qwen-Image LoRAs to the DiT in place (runtime
    /// adapter). Recommended for step-distillation LoRAs: the low-rank term is added in the
    /// activation path, so it survives bf16 (unlike `fuse` into a bf16 weight). Overhead is
    /// one extra rank-r matmul per adapted Linear — negligible against the 20B base.
    public static func apply(
        diffusersLoRAs loras: [(url: URL, strength: Float)],
        to model: QwenImageTransformer2DModel,
        dtype: DType = .bfloat16
    ) throws {
        let (params, ranks, targetKeys) = try combined(loras, dtype: dtype)
        replaceTargets(in: model, keys: targetKeys) { path, linear in
            ranks[path].map { LoRALinear.from(linear: linear, rank: $0, scale: 1.0) }
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
    }

    /// Convenience: apply a single LoRA at `strength` (1.0 = the documented diffusers
    /// `load_lora_weights` default).
    public static func apply(
        diffusersLoRA url: URL,
        to model: QwenImageTransformer2DModel,
        dtype: DType = .bfloat16,
        strength: Float = 1.0
    ) throws {
        try apply(diffusersLoRAs: [(url, strength)], to: model, dtype: dtype)
    }

    /// Apply, then permanently fuse a single adapter into the base Linears (W += B·A).
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
        dtype: DType = .bfloat16,
        strength: Float = 1.0
    ) throws {
        let (params, ranks, targetKeys) = try combined([(url, strength)], dtype: dtype)
        replaceTargets(in: model, keys: targetKeys) { path, linear in
            ranks[path].map { LoRALinear.from(linear: linear, rank: $0, scale: 1.0) }
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
        // Collapse every LoRALinear back into a plain fused Linear.
        for block in model.transformerBlocks {
            var fused: [(String, Module)] = []
            for (key, child) in block.namedModules() where targetKeys.contains(key) {
                if let lora = child as? LoRALayer { fused.append((key, lora.fused())) }
            }
            if !fused.isEmpty { block.update(modules: .unflattened(fused)) }
        }
    }

    /// Replace each targeted leaf `Linear` in every block with a transformed module. The
    /// transform receives the full `transformer_blocks.<i>.<rel>` path (so it can look up a
    /// per-layer rank) and may return nil to skip.
    private static func replaceTargets(
        in model: QwenImageTransformer2DModel,
        keys: Set<String>,
        _ transform: (_ path: String, _ linear: Linear) -> Module?
    ) {
        for (i, block) in model.transformerBlocks.enumerated() {
            var update: [(String, Module)] = []
            for (key, child) in block.namedModules() where keys.contains(key) {
                if let linear = child as? Linear,
                    let m = transform("transformer_blocks.\(i).\(key)", linear) {
                    update.append((key, m))
                }
            }
            if !update.isEmpty { block.update(modules: .unflattened(update)) }
        }
    }
}

/// Stateful LoRA hot-swapper for a resident DiT: switch the active adapter set (e.g. an
/// effect picked from a dropdown) WITHOUT reloading the 20B base.
///
/// `set(_:)` first detaches the current adapter — restoring the pristine base modules
/// captured on first use — then applies the new combo. Because `LoRALinear` shares the base
/// weight `MLXArray` by reference (`super.init(weight:bias:)`), neither the capture nor a
/// swap duplicates base weights: only the small `lora_a`/`lora_b` factors are added and freed
/// (≈ one rank-r pair per adapted Linear). Re-attaching is required because `LoRALinear`
/// IS-A `Linear`, so a naive second `apply()` would wrap a LoRALinear in another LoRALinear.
///
/// Not thread-safe; drive it from the same actor that owns the DiT.
public final class QwenImageEditLoRASwapper {
    private let model: QwenImageTransformer2DModel
    private let dtype: DType
    /// full `transformer_blocks.<i>.<rel>` path -> pristine base module (Linear/QuantizedLinear).
    private var pristine: [String: Module] = [:]
    /// block-relative keys currently realized as a LoRALinear.
    private var applied: Set<String> = []

    public init(model: QwenImageTransformer2DModel, dtype: DType = .bfloat16) {
        self.model = model
        self.dtype = dtype
    }

    /// The block-relative keys currently adapted (empty = pure base).
    public var activeKeys: Set<String> { applied }

    /// Make `loras` the active adapter set. An empty array leaves the pristine base in place.
    /// Diffusers/kohya dialects are handled by `QwenImageEditLoRA` (see `combined`).
    public func set(_ loras: [(url: URL, strength: Float)]) throws {
        detach()
        guard !loras.isEmpty else { return }
        let (params, ranks, targetKeys) = try QwenImageEditLoRA.combined(loras, dtype: dtype)
        for (i, block) in model.transformerBlocks.enumerated() {
            var update: [(String, Module)] = []
            for (key, child) in block.namedModules() where targetKeys.contains(key) {
                guard let linear = child as? Linear, let rank = ranks["transformer_blocks.\(i).\(key)"]
                else { continue }
                // First time we touch this target, the child IS the pristine base — capture it
                // (shares weights by reference, so capture is free) so detach can restore it.
                pristine["transformer_blocks.\(i).\(key)"] = pristine["transformer_blocks.\(i).\(key)"] ?? linear
                update.append((key, LoRALinear.from(linear: linear, rank: rank, scale: 1.0)))
            }
            if !update.isEmpty { block.update(modules: .unflattened(update)) }
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
        applied = targetKeys
    }

    /// Restore the pristine base in every currently-adapted target.
    public func detach() {
        guard !applied.isEmpty else { return }
        for (i, block) in model.transformerBlocks.enumerated() {
            var restore: [(String, Module)] = []
            for key in applied {
                if let orig = pristine["transformer_blocks.\(i).\(key)"] { restore.append((key, orig)) }
            }
            if !restore.isEmpty { block.update(modules: .unflattened(restore)) }
        }
        applied = []
    }
}
