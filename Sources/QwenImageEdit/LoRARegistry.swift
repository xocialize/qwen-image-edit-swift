// Shared LoRA registry / cache infrastructure for every wrapper over this core.
//
// These types started in MLXQwenImageEditTurbo (edit tier, one effect LoRA per request) and
// moved here when the Flash T2I tier needed the same machinery: a manifest of community
// adapters and lazy per-adapter download+cache. Nothing here is edit- or T2I-specific:
// `LoRAEntry` is just "a repo + a file inside it", which serves the curated registry, our
// mirrored gallery, and a user-pasted custom repo identically. The Turbo target keeps
// `LoRARegistry.bundled()` because that reads ITS bundled resource; everything else is common.
//
// Structural compatibility across the diffusers / diffusion_model / lora.down-up / kohya
// dialects is handled by QwenImageEditLoRA — see QwenImageEditLoRASwapper.
//
// Deliberately Foundation-only: this core is a standalone inference port and does NOT depend
// on MLXToolKit (the engine contract). The `metaData` half of the LoRA story — how a REQUEST
// selects adapters — therefore lives in the engine wrappers, not here.

import Foundation

/// One selectable adapter. `repo`/`weightFile` resolve to a HF `resolve/main` download URL.
public struct LoRAEntry: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let repo: String
    public let weightFile: String
    public let defaultStrength: Float
    public let trigger: String

    public init(
        id: String, displayName: String, repo: String, weightFile: String,
        defaultStrength: Float = 1.0, trigger: String = ""
    ) {
        self.id = id
        self.displayName = displayName
        self.repo = repo
        self.weightFile = weightFile
        self.defaultStrength = defaultStrength
        self.trigger = trigger
    }
}

/// The decoded manifest plus id lookup.
public struct LoRARegistry: Codable, Sendable {
    public let schemaVersion: Int
    public let base: String
    public let adapters: [LoRAEntry]

    public init(schemaVersion: Int, base: String, adapters: [LoRAEntry]) {
        self.schemaVersion = schemaVersion
        self.base = base
        self.adapters = adapters
    }

    public func entry(id: String) -> LoRAEntry? { adapters.first { $0.id == id } }
}

public enum LoRARegistryError: Error, LocalizedError {
    case manifestMissing
    case unknownAdapter(String)
    case download(String, underlying: String)

    public var errorDescription: String? {
        switch self {
        case .manifestMissing: return "Bundled LoRA registry manifest not found."
        case .unknownAdapter(let id): return "No LoRA with id '\(id)' in the registry."
        case .download(let id, let underlying):
            return "Failed to download LoRA '\(id)': \(underlying)"
        }
    }
}

/// Lazy-downloads + caches adapters from HuggingFace `resolve/main`.
public struct LoRACache: Sendable {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    /// Local path for an entry's cached file (id-named so display order / weightFile renames
    /// don't fork the cache).
    public func localURL(for entry: LoRAEntry) -> URL {
        directory.appendingPathComponent("\(entry.id).safetensors")
    }

    /// Return the cached file, downloading it on first use. Atomic (download to a temp file,
    /// then move) so a partial download never poisons the cache.
    public func ensure(_ entry: LoRAEntry) async throws -> URL {
        let dest = localURL(for: entry)
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        // Percent-encode the path so non-ASCII / spaced weight names (e.g. 镜头转换.safetensors,
        // "Style Transfer-Alpha-V0.1.safetensors") resolve.
        comps.path = "/\(entry.repo)/resolve/main/\(entry.weightFile)"
        guard let url = comps.url else {
            throw LoRARegistryError.download(entry.id, underlying: "bad URL for \(entry.weightFile)")
        }
        do {
            let (tmp, response) = try await URLSession.shared.download(from: url)
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw LoRARegistryError.download(entry.id, underlying: "HTTP \(http.statusCode)")
            }
            // Move into place (replace any stale temp at dest).
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tmp, to: dest)
            return dest
        } catch is CancellationError {
            // CAN-2: never launder a cancellation into LoRARegistryError — the engine's lane
            // disambiguation and the caller's .cancelled classification key on the type.
            throw CancellationError()
        } catch let e as LoRARegistryError {
            throw e
        } catch {
            throw LoRARegistryError.download(entry.id, underlying: error.localizedDescription)
        }
    }
}
