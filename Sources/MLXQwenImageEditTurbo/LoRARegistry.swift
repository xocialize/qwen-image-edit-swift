// The turbo tier's bundled effect registry.
//
// `LoRAEntry` / `LoRARegistry` / `LoRACache` / `LoRAMetaKeys` moved into the shared
// `QwenImageEdit` core when the Flash T2I tier needed the same machinery — they are
// re-exported here so existing `import MLXQwenImageEditTurbo` call sites keep compiling.
// What stays target-local is `bundled()`, which reads THIS target's resource.

import Foundation
import MLXToolKit
import QwenImageEdit

@_exported import struct QwenImageEdit.LoRAEntry
@_exported import struct QwenImageEdit.LoRARegistry
@_exported import struct QwenImageEdit.LoRACache
@_exported import enum QwenImageEdit.LoRARegistryError

/// `metaData` keys the turbo package reads for per-request effect selection.
public enum LoRAMetaKeys {
    /// Registry id of the effect to apply (absent / empty = base turbo, Lightning only).
    public static let id = "loraId"
    /// Optional strength override; defaults to the entry's `defaultStrength`.
    public static let strength = "loraStrength"
}

extension MetaValue {
    /// String payload (accepts a JSON string).
    public var asString: String? { if case .string(let s) = self { return s }; return nil }
    /// Numeric payload as Float (accepts int or double).
    public var asFloat: Float? {
        switch self {
        case .double(let d): return Float(d)
        case .int(let i): return Float(i)
        default: return nil
        }
    }
}

extension LoRARegistry {
    /// Load the manifest bundled with this target (the curated edit-tier effect list).
    public static func bundled() throws -> LoRARegistry {
        guard let url = Bundle.module.url(forResource: "qie-lora-registry", withExtension: "json")
        else { throw LoRARegistryError.manifestMissing }
        return try JSONDecoder().decode(LoRARegistry.self, from: Data(contentsOf: url))
    }
}
