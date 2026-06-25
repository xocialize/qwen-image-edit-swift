// Pre-quantized DiT: convert the bf16 transformer to a single int4/int8 safetensors ONCE
// (at the bf16 peak, on a big-RAM box), then load it on consumers with NO bf16 peak.
//
// The load trick: build the model (lazy placeholder weights — cheap, never materialized),
// set up the QuantizedLinear structure with the same filter, then load the pre-quantized
// weights straight in. The bf16 placeholders are dropped by update(), so the only memory
// realized is the int4 weights (~10 GB) — versus ~40 GB to load bf16 then quantize.

import Foundation
import MLX
import MLXNN

extension QwenImageEditWeights {

    /// Which DiT Linears to quantize and at what precision. attn + feed-forward at
    /// `ditBits`; the conditioning-critical modulation linears at `modulationBits` (nil =
    /// leave full precision; int4 there is grainy). The filter and metadata are shared by
    /// the convert and load paths so a quantized file is self-describing.
    public struct DiTQuantConfig: Sendable {
        public var ditBits: Int
        public var modulationBits: Int?
        public var groupSize: Int

        public init(ditBits: Int, modulationBits: Int? = nil, groupSize: Int = 64) {
            self.ditBits = ditBits
            self.modulationBits = modulationBits
            self.groupSize = groupSize
        }

        func spec(_ path: String, _ module: Module)
            -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
        {
            guard module is Linear, path.contains("transformer_blocks") else { return nil }
            if path.contains(".attn.") || path.contains("_mlp.") {
                return (groupSize, ditBits, .affine)
            }
            if let mb = modulationBits, path.contains(".img_mod") || path.contains(".txt_mod") {
                return (groupSize, mb, .affine)
            }
            return nil
        }

        var metadata: [String: String] {
            [
                "dit_bits": "\(ditBits)", "group_size": "\(groupSize)",
                "modulation_bits": modulationBits.map { "\($0)" } ?? "none",
            ]
        }

        static func from(metadata m: [String: String]) throws -> DiTQuantConfig {
            guard let db = m["dit_bits"].flatMap(Int.init),
                let gs = m["group_size"].flatMap(Int.init)
            else { throw QwenImageEditError.loading("quantized DiT: missing quant metadata") }
            let mb = m["modulation_bits"].flatMap { $0 == "none" ? nil : Int($0) }
            return DiTQuantConfig(ditBits: db, modulationBits: mb, groupSize: gs)
        }
    }

    /// One-time conversion: bf16 diffusers `transformer/` -> one self-describing
    /// pre-quantized safetensors. Runs at the bf16 peak — do it once on a big-RAM box.
    public static func saveQuantizedDiT(
        from ptDir: URL, to outURL: URL, config: DiTQuantConfig
    ) throws {
        let model = try loadDiTFromPT(directory: ptDir, dtype: .bfloat16)
        quantize(model: model, filter: config.spec)
        eval(model)
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: params, metadata: config.metadata, url: outURL)
    }

    /// Load a pre-quantized DiT with no bf16 peak (see file header). Self-describing: the
    /// quant config is read from the file's metadata.
    public static func loadQuantizedDiT(from url: URL) throws -> QwenImageTransformer2DModel {
        let (weights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let config = try DiTQuantConfig.from(metadata: metadata)
        let model = QwenImageTransformer2DModel()
        quantize(model: model, filter: config.spec)  // structure only; placeholders stay lazy
        try verifyAndLoad(model: model, weights: weights, label: "DiT(int\(config.ditBits))")
        return model
    }
}
