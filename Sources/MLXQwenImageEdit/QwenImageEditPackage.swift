// MLXEngine `imageEdit` package over the QwenImageEdit core — the engine's first
// imageEdit surface (contract 1.2.0).
//
// Qwen-Image-Edit-2511 (Apache-2.0): Qwen2.5-VL-7B-conditioned 20B zero_cond_t DiT
// + Wan 3D causal VAE. The Swift core is parity-locked against the P2 PT goldens
// (DiT 0.99986 · VAE decode 73.7 dB · VAE encode 1.0 · encoder 0.9977 fp32); this
// wrapper is a thin conformance layer — all model logic lives in `QwenImageEdit`.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXProfiling
import MLXToolKit
import QwenImageEdit
import UniformTypeIdentifiers

/// Init-time configuration (C9): the 2511 snapshot root and generation defaults.
public struct QwenImageEditConfiguration:
    PackageConfiguration, ModelStorable, QuantConfigured, WeightSourcing
{
    /// Explicit snapshot root with `transformer/`, `vae/`, `text_encoder/`, `processor/`.
    /// Empty = resolve from the model store, materializing from `repo` on first run.
    public var snapshotPath: String
    public var defaultSteps: Int
    public var defaultTrueCFGScale: Float
    /// Default step-residual cache level for the denoise loop (FR2 — DBCache family;
    /// see QwenImageEdit.StepCacheMode). nil/.off = full compute every step. Requests
    /// can override via metaData `stepCacheMode`. Opt-in until the SSIMULACRA2 ≥85
    /// promotion gate passes.
    public var stepCache: StepCacheMode?
    public var modelsRootDirectory: URL?

    /// Always bf16 (the only declared tier for the base package). Lets the memory governor
    /// charge the matching split `QuantFootprint` instead of the largest-that-fits guess.
    public var quant: Quant { .bf16 }

    /// Upstream weights. Unlike the Flash tier there is no MLX mirror to publish: this port
    /// reads the diffusers-layout safetensors directly (key sanitizing + conv transposes happen
    /// at load), and upstream already ships `processor/tokenizer.json` — the file whose absence
    /// forced a mirror for Flash. So the materializer points straight at Qwen.
    public static let repo = "Qwen/Qwen-Image-Edit-2511"

    /// Fresh-machine sources (MAT), split by role so a future quantized tier can swap the
    /// transformer glob without touching the rest.
    public var weightSources: [WeightSource] {
        [
            WeightSource(
                role: "transformer", repo: Self.repo, revision: "main",
                matching: ["transformer/*"]),
            WeightSource(role: "vae", repo: Self.repo, revision: "main", matching: ["vae/*"]),
            WeightSource(
                role: "text-encoder", repo: Self.repo, revision: "main",
                matching: ["text_encoder/*", "processor/*"]),
            WeightSource(
                role: "pipeline-config", repo: Self.repo, revision: "main",
                matching: ["model_index.json", "scheduler/*"]),
        ]
    }

    /// Honor an explicit snapshot path first (it satisfies everything), then the store layout.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if !snapshotPath.isEmpty,
            FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: snapshotPath)
                    .appendingPathComponent("transformer").path)
        {
            return []
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// Store-resolved snapshot root (what `load()` uses after materialization): explicit path
    /// wins, then the engine-executed flat layout, then a hub-client `snapshots/<commit>/`.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        let fm = FileManager.default
        if let flat = store.directory(for: Self.repo),
            fm.fileExists(atPath: flat.appendingPathComponent("transformer").path)
        {
            return flat
        }
        if let snap = store.snapshotDirectory(for: Self.repo, revision: "main"),
            fm.fileExists(atPath: snap.appendingPathComponent("transformer").path)
        {
            return snap
        }
        return store.directory(for: Self.repo)
    }

    public init(
        /// Empty (the default) routes through the model store; pass a path to pin a local snapshot.
        snapshotPath: String = "",
        defaultSteps: Int = 20,
        defaultTrueCFGScale: Float = 4.0,
        stepCache: StepCacheMode? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.stepCache = stepCache
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, defaultSteps, defaultTrueCFGScale, stepCache
    }
}

public enum QwenImageEditPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "2511 snapshot not readable at \(p)."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class QwenImageEditPackage: ModelPackage {
    public typealias Configuration = QwenImageEditConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "Qwen/Qwen-Image-Edit-2511", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (efficiency contract 1.14.0). Per-stage eviction (P2): the
                // VL-7B encoder loads per request and is evicted before the denoise peak, so it
                // is a TRANSIENT, not a resident. Measured M5 Max, 1024²/8-step CFG4 (QIE_MEMBENCH
                // in PackageTests): resident floor (DiT bf16 40.9 GB + fp32 VAE 0.5 GB) = 41.4 GB;
                // worst peak 59.2 GB → activation (peak − floor, dominated by the ~16.6 GB transient
                // encoder load during encode) = 17.9 GB → 21 GB at +20% headroom. Old flat declaration
                // folded the encoder into resident (60 GB); the split lets two base models co-reside
                // under ONE shared activation reserve. 4-bit DiT+VL is the Turbo int4 tier.
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 42_000_000_000,
                        peakActivationBytes: 21_000_000_000)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                IEditContract.descriptor(
                    name: "qwen-image-edit",
                    summary: "Qwen-Image-Edit-2511 instruction editing (20B zero_cond_t "
                        + "DiT + Qwen2.5-VL conditioning): multi-image fusion, identity-"
                        + "preserving edits, 1024²-area output, 20-step true CFG. Opt-in "
                        + "step-residual cache via metaData stepCacheMode "
                        + "(conservative|aggressive).",
                    modes: []
                )
            ]
        )
    }

    private let configuration: Configuration
    private var generator: QwenImageEditGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        // First-run materialization is engine-executed (contract 1.24): the declared
        // WeightSourcing sources are fetched before load() runs. This guard is the offline
        // backstop so absent weights still fail legibly.
        guard let snapshot = configuration.resolvedSnapshotDirectory(
            storeRoot: configuration.modelsRootDirectory),
            FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            throw QwenImageEditPackageError.unreadableSnapshot(
                configuration.snapshotPath.isEmpty
                    ? QwenImageEditConfiguration.repo : configuration.snapshotPath)
        }

        // DiT + VAE stay resident; the VL-7B encoder is loaded per request and evicted
        // before the denoise peak (efficiency contract 1.14.0 — see QwenImageEditGenerator).
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImageEditGenerator(
            encoderProvider: { try await QwenVLPromptEncoder.load(snapshot: snapshot) },
            transformer: transformer, vae: vae)
        // FR4: absorb first-forward graph/kernel build at load, not on the first edit.
        generator.warmup()
        self.generator = generator
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence lives in the shared core (QwenImageEditGenerator
        // .generate: post-encode seam, per-denoise-step checkpoint, pre-decode seam — all
        // `try Task.checkCancellation()`, rethrown unchanged through this throwing seam).
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .imageEdit, let edit = request as? IEditRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        guard !edit.images.isEmpty else {
            throw QwenImageEditPackageError.imageDecode
        }
        // Multi-image fusion: decode all conditioning images in prompt order
        // ("Picture 1", "Picture 2", …); the core packs per-image VAE latents + grids.
        try Task.checkCancellation()
        let inputs = try edit.images.map { try Self.decodeRGB($0.data) }

        // Run-level MLX profiling (MLX_PROFILE=1): stage spans live in the core; the run
        // summary normalizes to ms/step. Output size = the core's 1024²-area rule.
        let steps = edit.steps ?? configuration.defaultSteps
        let (tw, th) = QwenVLPromptEncoder.calculateDimensions(
            targetArea: 1024 * 1024,
            ratio: Double(inputs[0].width) / Double(inputs[0].height))
        let prof = MLXProfiler.shared
        prof.beginRun("qwen-image-edit imageEdit steps=\(steps) \(tw)x\(th)")

        // Step-residual cache: request metaData overrides the configured default (opt-in).
        var stepCache = configuration.stepCache ?? .off
        if case .string(let raw)? = edit.metaData[StepCacheMetaKeys.mode],
            let mode = StepCacheMode(rawValue: raw)
        {
            stepCache = mode
        }
        let (pixels, w, h) = try await generator.generate(
            images: inputs,
            prompt: edit.prompt,
            negativePrompt: edit.negativePrompt ?? " ",
            steps: steps,
            trueCFGScale: edit.guidanceScale.map(Float.init)
                ?? configuration.defaultTrueCFGScale,
            seed: edit.seed ?? 0,
            stepCache: stepCache,
            progress: { _, _ in })
        prof.endRun(denominators: ["step": Double(steps)])

        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return IEditResponse(image: Image(format: .png, data: png, width: w, height: h))
    }

    /// PNG/JPEG Data -> interleaved RGB8 in sRGB.
    nonisolated static func decodeRGB(_ data: Data) throws
        -> (rgb: [UInt8], width: Int, height: Int)
    {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw QwenImageEditPackageError.imageDecode }
        let w = cg.width
        let h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw QwenImageEditPackageError.imageDecode }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return (rgb, w, h)
    }

    /// Interleaved RGB8 -> PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw QwenImageEditPackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw QwenImageEditPackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImageEditPackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw QwenImageEditPackageError.pngEncode
        }
        return out as Data
    }
}

extension QwenImageEditPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(QwenImageEditPackage.self)
    }
}
