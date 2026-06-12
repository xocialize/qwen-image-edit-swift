// Weight loading for the Qwen-Image-Edit-2511 DiT.
//
// The diffusers `transformer/` checkpoint is pure Linear + RMSNorm (block LayerNorms
// are affine-less — no weights). PT<->MLX layouts identical, no transposes. Renames:
//   .img_mod.1. / .txt_mod.1.  ->  .img_mod. / .txt_mod.   (Sequential(SiLU, Linear))
//   .net.0.proj. / .net.2.     ->  .proj_in. / .proj_out.  (FeedForward GELU stack)

import Foundation
import MLX
import MLXNN

public enum QwenImageEditWeights {

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw QwenImageEditError.loading("no .safetensors under \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            merged.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        return merged
    }

    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: ".img_mod.1.", with: ".img_mod.")
            .replacingOccurrences(of: ".txt_mod.1.", with: ".txt_mod.")
            .replacingOccurrences(of: ".net.0.proj.", with: ".proj_in.")
            .replacingOccurrences(of: ".net.2.", with: ".proj_out.")
    }

    /// Load the DiT from the original diffusers `transformer/` safetensors.
    public static func loadDiTFromPT(directory: URL, dtype: DType = .bfloat16) throws
        -> QwenImageTransformer2DModel
    {
        let model = QwenImageTransformer2DModel()
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) {
            weights[sanitizeDiTKey(k)] = v.asType(dtype)
        }
        try verifyAndLoad(model: model, weights: weights, label: "DiT(PT)")
        return model
    }

    /// Two-way strict load: all module keys filled AND every checkpoint key consumed —
    /// a partial load emits garbage with no other symptom.
    static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw QwenImageEditError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. "
                    + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw QwenImageEditError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. "
                    + unused.prefix(4).joined(separator: ", "))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }
}
