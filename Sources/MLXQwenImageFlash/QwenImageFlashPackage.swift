// MLXEngine `textToImage` package over the shared QwenImageEdit core — the 4-step tier.
//
// nvidia/Qwen-Image-Flash: a DMD2 distillation of Qwen/Qwen-Image that replaces the 20B
// DiT's weights and touches nothing else. The transformer weight-key set is IDENTICAL to
// Qwen-Image-Edit-2511's (1933 tensors), and the text encoder / tokenizer / VAE files are
// byte-identical (sha256-verified) — so this package reuses the parity-locked
// `QwenImageEdit` core wholesale and only drives its T2I path (`QwenImageT2IGenerator`).
//
// The distillation internalized CFG 4.0, so inference runs `true_cfg_scale = 1.0`: ONE DiT
// forward per step, four steps, on the packaged static shift-3 schedule
// ([1.0, 0.9, 0.75, 0.5, 0.0]). Deviating from that schedule or re-applying CFG double-counts
// guidance the student already absorbed.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXProfiling
import MLXToolKit
import QwenImageEdit
import UniformTypeIdentifiers

/// Init-time configuration (C9): where the snapshot lives + generation defaults.
public struct QwenImageFlashConfiguration:
    PackageConfiguration, ModelStorable, QuantConfigured, BudgetAware, WeightSourcing
{
    /// Explicit snapshot root (`transformer/`, `vae/`, `text_encoder/`, `tokenizer/`).
    /// `nil` = resolve from the model store, materializing on first run.
    public var snapshotPath: String?
    public var defaultSteps: Int
    /// 1.0 — guidance was internalized by the DMD2 distillation. Raising it re-applies CFG the
    /// student already absorbed AND doubles the DiT forwards per step.
    public var defaultTrueCFGScale: Float
    public var defaultWidth: Int
    public var defaultHeight: Int
    public var modelsRootDirectory: URL?

    /// Requested precision tier. `.bf16` is the quality reference; `.int8` is the device-reach
    /// tier (int8 attn+FFN+modulation DiT, int8 VL-7B text model, full-precision VAE). Both are
    /// PRE-quantized artifacts — see `effectiveQuant` for why this is not a load-time choice.
    ///
    /// int4 was built and REJECTED on measurement, not on principle: DiT step-0 cosine 0.9623
    /// at group 64 and 0.9659 at group 32 (vs 0.99836 bf16 / 0.9973 int8), and the 1024² render
    /// came out visibly soft and washed out with fine fur/snow detail gone. Finer scales did not
    /// rescue it — this DiT is intrinsically lossy at 4 bits. The recipe and its numbers live in
    /// FlashQuantSweepTests so the decision is reproducible rather than folklore.
    public var quant: Quant

    /// Set by the engine's memory governor before `resident()`/`prepare()` (BudgetAware).
    public var availableBudgetBytes: UInt64?

    /// The tier actually loaded. A declared `.bf16` degrades to `.int8` when the governor's
    /// budget cannot seat the bf16 tier, which is the whole point of publishing both: bf16
    /// needs ~61 GB (41.4 floor + 19.3 activation) and simply does not fit most Macs.
    ///
    /// This is resolved BEFORE materialization, not at load: `weightSources` keys off it too,
    /// so a budget-constrained machine downloads the int8 snapshot rather than fetching 41 GB
    /// of bf16 it can never seat.
    public var effectiveQuant: Quant {
        if quant != .bf16 { return quant }
        guard let budget = availableBudgetBytes else { return .bf16 }
        return budget >= Self.bf16TotalBytes ? .bf16 : .int8
    }

    /// bf16 resident floor + activation, from the measured split footprint.
    static let bf16TotalBytes: UInt64 = 41_400_000_000 + 19_300_000_000

    /// The published MLX snapshots, NOT the upstream repo. The weights are byte-identical to
    /// `nvidia/Qwen-Image-Flash` (verified by blob hash) — the reason to source from
    /// mlx-community is `tokenizer/tokenizer.json`: upstream ships slow-tokenizer files only
    /// (`vocab.json` + `merges.txt`), which swift-transformers cannot read, so a fresh machine
    /// pointed at upstream materializes a snapshot this package cannot load.
    public static let bf16Repo = "mlx-community/Qwen-Image-Flash-bf16"
    /// Pre-quantized tier: int8 DiT + int8 VL-7B text model, converted once at the bf16 peak.
    /// Consumers never pay a bf16 peak — which is what makes this tier reachable on machines
    /// that cannot hold the bf16 weights at all.
    public static let int8Repo = "mlx-community/Qwen-Image-Flash-8bit"

    /// Back-compat alias for the default (bf16) repo.
    public static let repo = bf16Repo

    public var effectiveRepo: String {
        effectiveQuant == .int8 ? Self.int8Repo : Self.bf16Repo
    }

    /// Pre-quantized file names inside an int8 snapshot.
    public static let int8DiTFile = "transformer/model-int8.safetensors"
    public static let int8TextEncoderFile = "text_encoder/model-int8.safetensors"

    public init(
        snapshotPath: String? = nil,
        quant: Quant = .bf16,
        defaultSteps: Int = 4,
        defaultTrueCFGScale: Float = 1.0,
        defaultWidth: Int = 1024,
        defaultHeight: Int = 1024,
        availableBudgetBytes: UInt64? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.quant = quant
        self.availableBudgetBytes = availableBudgetBytes
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.defaultWidth = defaultWidth
        self.defaultHeight = defaultHeight
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// Fresh-machine sources (MAT), split by role and keyed off `effectiveQuant` so the tier
    /// that will be LOADED is the tier that gets downloaded.
    public var weightSources: [WeightSource] {
        let repo = effectiveRepo
        // Quant-aware globs: the int8 tier must NOT pull the 41 GB bf16 transformer it will
        // never load (that download is the thing making the tier pointless on a small machine).
        let transformer = effectiveQuant == .int8
            ? [Self.int8DiTFile, "transformer/config.json"]
            : ["transformer/*"]
        let textEncoder = effectiveQuant == .int8
            ? [Self.int8TextEncoderFile, "text_encoder/config.json", "tokenizer/*"]
            : ["text_encoder/*", "tokenizer/*"]
        return [
            WeightSource(
                role: "transformer", repo: repo, revision: "main", matching: transformer),
            WeightSource(role: "vae", repo: repo, revision: "main", matching: ["vae/*"]),
            WeightSource(
                role: "text-encoder", repo: repo, revision: "main", matching: textEncoder),
            WeightSource(
                role: "pipeline-config", repo: repo, revision: "main",
                matching: ["model_index.json", "scheduler/*"]),
        ]
    }

    /// Honor the explicit snapshot path first (it satisfies everything), then the store layout.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let snapshotPath,
            FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: snapshotPath)
                    .appendingPathComponent("transformer").path)
        {
            return []
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// Store-resolved snapshot root (what `load()` uses after materialization): the explicit
    /// path wins, then the engine-executed flat layout, then a hub-client `snapshots/<commit>/`.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        let fm = FileManager.default
        if let flat = store.directory(for: effectiveRepo),
            fm.fileExists(atPath: flat.appendingPathComponent("transformer").path)
        {
            return flat
        }
        if let snap = store.snapshotDirectory(for: effectiveRepo, revision: "main"),
            fm.fileExists(atPath: snap.appendingPathComponent("transformer").path)
        {
            return snap
        }
        return store.directory(for: effectiveRepo)
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, quant, defaultSteps, defaultTrueCFGScale, defaultWidth, defaultHeight
    }
}

public enum QwenImageFlashPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Qwen-Image-Flash snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class QwenImageFlashPackage: ModelPackage {
    public typealias Configuration = QwenImageFlashConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: NVIDIA Open Model License — commercially permissive (allowlisted in
            // MLXToolKit with the review rationale). §3.1 obligation: any redistribution of
            // these WEIGHTS must ship a copy of the Agreement plus a Notice file reading
            // "Licensed by NVIDIA Corporation under the NVIDIA Open Model License". The repo
            // additionally carries Apache-2.0 as "Additional Information". C8: port code MIT.
            license: LicenseDeclaration(
                weightLicense: .nvidiaOpenModel, portCodeLicense: .mit),
            // Provenance points at the ORIGINAL model; `Configuration.repo` is where the
            // materializer fetches from (the mlx-community mirror carrying tokenizer.json).
            provenance: Provenance(
                sourceRepo: "nvidia/Qwen-Image-Flash", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (efficiency contract 1.14.0), MEASURED by this package's own
                // mem-bench (FlashMemBenchTests, QIF_MEMBENCH=1) on M5 Max at the input
                // envelope — 1024², 4 steps, true CFG 1.0:
                //   resident floor (DiT bf16 + fp32 VAE, post-load, cache cleared) = 41.4 GB
                //   worst peak                                                     = 57.4 GB
                //   activation (peak − floor)                                      = 16.1 GB
                //                                            → 19.3 GB at +20% headroom
                // The VL-7B encoder is a TRANSIENT, not a resident: it loads per request and is
                // evicted before the denoise peak, and its ~16.6 GB load is what dominates the
                // activation term. T2I comes in under the edit path's measured 17.9 GB because
                // it has no conditioning tokens — the DiT sequence is the target grid alone.
                // [Smoke MLX-peak, not in-app phys_footprint; re-baseline once registered in
                //  the image app (the BiRefNet ~2.7× lesson).]
                //
                // int8 (FlashQuantGateTests, same envelope): floor 22.3 GB, peak 30.0 GB,
                // activation 7.8 GB → 9.3 GB at +20%. Near-lossless (DiT step-0 cos 0.9973 vs
                // bf16's 0.99836; encoder 0.99992) and 4× faster end-to-end — 19.8 s vs 83.3 s
                // at 1024²/4 steps, with a 2.3 s load instead of ~60 s. This is the tier that
                // makes the model reachable below a 128 GB machine.
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 41_400_000_000,
                        peakActivationBytes: 19_300_000_000),
                    QuantFootprint(
                        quant: .int8, residentBytes: 22_300_000_000,
                        peakActivationBytes: 9_300_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "qwen-image-flash",
                    summary: "Qwen-Image-Flash text-to-image (NVIDIA DMD2 4-step distill of "
                        + "Qwen-Image, 20B MMDiT + Qwen2.5-VL conditioning): four deterministic "
                        + "steps on a static shift-3 FlowMatch Euler schedule with guidance "
                        + "internalized (true CFG 1.0), tested at 1024²; any size divisible "
                        + "by 16. Optional multi-LoRA stack per request (metaData `loras`: "
                        + "[{id, repo, file, strength}]), hot-swapped on the resident DiT and "
                        + "lazy-cached from HuggingFace.",
                    modes: []
                )
            ]
        )
    }

    private let configuration: Configuration
    private var generator: QwenImageT2IGenerator?
    /// Community-LoRA stack: adapters rank-stack on the resident 20B DiT via the swapper,
    /// so switching styles costs only the rank-r factors — never a base reload.
    private var swapper: QwenImageEditLoRASwapper?
    private var loraCache: LoRACache?
    /// Last applied stack, so a re-roll (new seed, same styles) skips the swap entirely.
    private var appliedStack: [LoRAStackItem] = []

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
            throw QwenImageFlashPackageError.unreadableSnapshot(
                configuration.snapshotPath ?? QwenImageFlashConfiguration.repo)
        }

        // DiT + VAE stay resident; the VL-7B encoder loads per request and is evicted before
        // the denoise peak. Text-only load: the ViT is never built (T2I conditions on text
        // alone), which also means a T2I snapshot needs no `processor/`.
        //
        // int8 loads PRE-quantized artifacts — never bf16-then-quantize. The bf16 peak exists
        // only in the one-time conversion (FlashQuantConvertTests), so this tier is loadable
        // on machines that could not hold the bf16 weights at all.
        let transformer: QwenImageTransformer2DModel
        let encoderQuantPath: String?
        if configuration.effectiveQuant == .int8 {
            let ditURL = snapshot.appendingPathComponent(Configuration.int8DiTFile)
            let encURL = snapshot.appendingPathComponent(Configuration.int8TextEncoderFile)
            guard FileManager.default.fileExists(atPath: ditURL.path),
                FileManager.default.fileExists(atPath: encURL.path)
            else { throw QwenImageFlashPackageError.unreadableSnapshot(ditURL.path) }
            transformer = try QwenImageEditWeights.loadQuantizedDiT(from: ditURL)
            encoderQuantPath = encURL.path
        } else {
            transformer = try QwenImageEditWeights.loadDiTFromPT(
                directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
            encoderQuantPath = nil
        }
        // VAE stays full precision at every tier: 0.25 GB on disk, and decode is where
        // precision loss shows up as visible colour/banding artifacts.
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        let shift = Self.readSchedulerShift(snapshot: snapshot)
        let generator = QwenImageT2IGenerator(
            encoderProvider: {
                try await QwenVLPromptEncoder.loadTextOnly(
                    snapshot: snapshot, quantizedTextModelPath: encoderQuantPath)
            },
            transformer: transformer, vae: vae, shift: shift)
        // Community-LoRA seam. The swapper holds the base weights by reference, so a style
        // change re-applies only the low-rank factors — the 20B DiT is never reloaded.
        // Runtime `apply` (what the swapper does), never `fuse`: at bf16/int8 a fused delta
        // rounds away below the ULP (see QwenImageEditLoRA's precision note).
        self.swapper = QwenImageEditLoRASwapper(model: transformer)
        let cacheRoot = configuration.modelsRootDirectory
            ?? (try? FileManager.default.url(
                for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        self.loraCache = LoRACache(directory: cacheRoot.appendingPathComponent("qif-lora-cache"))
        self.appliedStack = []
        // FR4: absorb the first-forward graph/kernel build at load, not on the first request.
        generator.warmup()
        self.generator = generator
    }

    public func unload() async {
        generator = nil
        swapper = nil
        loraCache = nil
        appliedStack = []
        MLX.Memory.clearCache()  // release the retained pool so eviction actually frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(), before notLoaded validation.
        // Mid-run cadence lives in the shared core (QwenImageT2IGenerator.generate: post-encode
        // seam, per-denoise-step checkpoint, pre-decode seam) and rethrows CancellationError
        // unchanged through this throwing seam.
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let (width, height) = QwenImageT2IGenerator.roundedSize(
            width: t2i.width ?? configuration.defaultWidth,
            height: t2i.height ?? configuration.defaultHeight)
        let steps = t2i.steps ?? configuration.defaultSteps
        let trueCFG = t2i.guidanceScale.map(Float.init) ?? configuration.defaultTrueCFGScale

        // LoRA stack: apply before the denoise loop, skipping the swap when the selection is
        // unchanged (a re-roll with a new seed is the common case). Trigger words are the
        // caller's job — they belong in the prompt, which the package does not rewrite.
        try await applyLoRAStack(LoRAStackItem.decode(t2i.metaData[FlashLoRAMetaKeys.stack]))

        let prof = MLXProfiler.shared
        prof.beginRun("qwen-image-flash textToImage steps=\(steps) \(width)x\(height)")
        let (pixels, w, h) = try await generator.generate(
            prompt: t2i.prompt,
            negativePrompt: t2i.negativePrompt ?? " ",
            width: width, height: height,
            steps: steps,
            trueCFGScale: trueCFG,
            seed: t2i.seed ?? 0,
            progress: { _, _ in })
        prof.endRun(denominators: ["step": Double(steps)])

        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))
    }

    /// Rank-stack the requested adapters on the resident DiT.
    ///
    /// No-op when the selection is unchanged, so re-rolling a seed costs nothing. An empty
    /// stack detaches back to the base weights. Downloads are lazy and cached per adapter id;
    /// a `CancellationError` from the download propagates unchanged (CAN-2).
    private func applyLoRAStack(_ stack: [LoRAStackItem]) async throws {
        guard let swapper else { return }
        guard stack != appliedStack else { return }
        guard !stack.isEmpty else {
            swapper.detach()
            appliedStack = []
            return
        }
        guard let loraCache else { return }
        var resolved: [(url: URL, strength: Float)] = []
        resolved.reserveCapacity(stack.count)
        for item in stack {
            resolved.append((try await loraCache.ensure(item.entry), item.strength))
        }
        try Task.checkCancellation()
        try swapper.set(resolved)
        appliedStack = stack
    }

    /// Static `shift` from the snapshot's packaged scheduler config (Flash: 3.0, dynamic off).
    nonisolated static func readSchedulerShift(snapshot: URL) -> Float {
        struct SchedCfg: Codable { var shift: Float? }
        let url = snapshot.appendingPathComponent("scheduler/scheduler_config.json")
        guard let data = try? Data(contentsOf: url),
            let cfg = try? JSONDecoder().decode(SchedCfg.self, from: data)
        else { return 3.0 }
        return cfg.shift ?? 3.0
    }

    /// Interleaved RGB8 -> PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw QwenImageFlashPackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw QwenImageFlashPackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImageFlashPackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw QwenImageFlashPackageError.pngEncode
        }
        return out as Data
    }
}

extension QwenImageFlashPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(QwenImageFlashPackage.self)
    }
}
