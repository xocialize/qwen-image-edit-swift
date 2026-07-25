// The T2I tier's multi-LoRA request contract.
//
// This lives in the WRAPPER, not the core: it is expressed in `MLXToolKit.MetaValue`, and the
// `QwenImageEdit` core is deliberately engine-contract-free (Foundation + MLX only) so it can
// be consumed without the engine. The core owns the transport-agnostic half — `LoRAEntry`,
// `LoRACache`, the dialect remapping in `QwenImageEditLoRA`.
//
// Why full coordinates instead of a registry id (which is what the edit tier's
// `metaData["loraId"]` carries): the gallery hands the user four slots, and a slot may point
// at ANY HuggingFace repo they paste. A curated-registry id cannot name those, so the request
// carries `{id, repo, file, strength}` per slot and the package resolves each through the cache.

import Foundation
import MLXToolKit
import QwenImageEdit

/// One entry of a multi-LoRA stack, as carried in `metaData[FlashLoRAMetaKeys.stack]`.
public struct LoRAStackItem: Sendable, Equatable {
    public let id: String
    public let repo: String
    public let file: String
    public let strength: Float

    public init(id: String, repo: String, file: String, strength: Float) {
        self.id = id
        self.repo = repo
        self.file = file
        self.strength = strength
    }

    /// Cache-facing view. `displayName`/`trigger` are presentation-only and unused here —
    /// trigger words are applied to the prompt by the caller, not the package.
    public var entry: LoRAEntry {
        LoRAEntry(
            id: id, displayName: id, repo: repo, weightFile: file,
            defaultStrength: strength, trigger: "")
    }

    /// Decode `[{id, repo, file, strength}, …]`. Malformed elements are skipped rather than
    /// failing the whole request — one bad slot should not cost the user a render.
    public static func decode(_ value: MetaValue?) -> [LoRAStackItem] {
        guard case .array(let items)? = value else { return [] }
        return items.compactMap { item in
            guard case .object(let o) = item,
                let repo = o.flashString("repo"), !repo.isEmpty,
                let file = o.flashString("file"), !file.isEmpty
            else { return nil }
            let id = o.flashString("id") ?? "\(repo)/\(file)"
            return LoRAStackItem(
                id: id, repo: repo, file: file, strength: o.flashFloat("strength") ?? 1.0)
        }
    }

    /// Encode a stack for a request's `metaData` (the app side of the same contract).
    public static func encode(_ items: [LoRAStackItem]) -> MetaValue {
        .array(items.map {
            .object([
                "id": .string($0.id), "repo": .string($0.repo), "file": .string($0.file),
                "strength": .double(Double($0.strength)),
            ])
        })
    }
}

/// `metaData` keys the Flash package reads.
public enum FlashLoRAMetaKeys {
    /// Multi-LoRA stack — array of `{id, repo, file, strength}`; absent/empty = base weights.
    public static let stack = "loras"
}

// Internal, not public: `MLXQwenImageEditTurbo` publishes its own `MetaValue.asString`/`asFloat`
// and an app importing both wrappers would get an ambiguous member lookup.
extension [String: MetaValue] {
    func flashString(_ key: String) -> String? {
        if case .string(let s)? = self[key] { return s }
        return nil
    }

    func flashFloat(_ key: String) -> Float? {
        switch self[key] {
        case .double(let d): return Float(d)
        case .int(let i): return Float(i)
        default: return nil
        }
    }
}
