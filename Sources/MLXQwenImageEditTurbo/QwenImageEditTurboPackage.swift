// MLXEngine fast/low-step `imageEdit` tier over the QwenImageEdit core.
//
// Same Qwen-Image-Edit-2511 (Apache-2.0) base as the standard package, but with the
// LightX2V Qwen-Image-Edit-2511-Lightning 4-step (DMD) LoRA (Apache-2.0) applied at
// RUNTIME via QwenImageEditLoRA.apply — NOT fused into a snapshot. Fusing this LoRA
// into bf16 weights rounds it away (per-weight deltas sit below bf16 ULP); runtime
// application adds the low-rank term in the activation path, where it survives bf16.
// Validated end to end (LoRAGenerateTests): coherent 4-step edits; effect scales
// linearly with `strength`.
//
// Surface: the canonical `imageEdit` capability with a `turbo` mode tag (per C4 — one
// editor, the tag declares the fast/DMD intent; not a new capability). Defaults: 4
// steps, true-CFG 1.0 (DMD single positive pass).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MLXToolKit
import QwenImageEdit
import UniformTypeIdentifiers

/// The `turbo` imageEdit mode (fast/low-step DMD variant).
public let turboMode = Mode(rawValue: "turbo")

/// Init-time configuration (C9): base 2511 snapshot + Lightning LoRA + DMD defaults.
public struct QwenImageEditTurboConfiguration: PackageConfiguration, ModelStorable {
    /// Base snapshot root with `transformer/`, `vae/`, `text_encoder/`, `processor/`.
    public var snapshotPath: String
    /// Diffusers-format Lightning 4-step LoRA (.safetensors), applied at load.
    public var loraPath: String
    /// LoRA strength multiplier on alpha/rank. 1.0 = the documented diffusers default, but
    /// the Lightning DMD is under-applied there in our pipeline; ~4 (0.5 effective) fully
    /// engages the 4-step distillation (strength eval) — hence the default below.
    public var strength: Float
    /// Quantize the DiT's attention + feed-forward Linears to this bit width before
    /// applying the LoRA (nil = bf16). 4 ≈ a ~4× smaller DiT; the LoRA rides the
    /// quantized base as QLoRALinear (its low-rank factors stay full precision).
    public var ditBits: Int?
    /// Quantize the VL-7B conditioning model to this bit width (nil = bf16). The biggest
    /// single non-DiT resident win toward the int4 tier.
    public var encoderBits: Int?
    /// Quantize the DiT modulation linears (img_mod/txt_mod) at this bit width (nil = keep
    /// full precision). They're the largest remaining bf16 chunk (~11 GB) but drive AdaLN
    /// conditioning: int4 here visibly degrades quality (graininess), so int8 is the safer
    /// footprint/quality tradeoff.
    public var modulationBits: Int?
    public var defaultSteps: Int
    public var defaultTrueCFGScale: Float
    public var modelsRootDirectory: URL?

    public init(
        snapshotPath: String =
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511",
        loraPath: String =
            "/Users/dustinnielson/Development/telestyle-work/loras/"
            + "QIE-2511-Lightning-4steps-V1.0-bf16.safetensors",
        strength: Float = 4.0,
        ditBits: Int? = nil,
        encoderBits: Int? = nil,
        modulationBits: Int? = nil,
        defaultSteps: Int = 4,
        defaultTrueCFGScale: Float = 1.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.loraPath = loraPath
        self.strength = strength
        self.ditBits = ditBits
        self.encoderBits = encoderBits
        self.modulationBits = modulationBits
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, loraPath, strength, ditBits, encoderBits, modulationBits
        case defaultSteps, defaultTrueCFGScale
    }
}

public enum QwenImageEditTurboPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case unreadableLoRA(String)
    case imageDecode
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "2511 base snapshot not readable at \(p)."
        case .unreadableLoRA(let p): return "Lightning LoRA not readable at \(p)."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class QwenImageEditTurboPackage: ModelPackage {
    public typealias Configuration = QwenImageEditTurboConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "lightx2v/Qwen-Image-Edit-2511-Lightning", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Measured resident (1024², runtime LoRA adds only rank factors):
                //   bf16: ~57 GB (20B DiT + bf16 VL-7B + fp32 VAE).
                //   int4 (ditBits=4, encoderBits=4, modulationBits=8): ~22 GB, quality
                //     intact — attn/mlp int4, conditioning-critical modulation int8 (int4
                //     there is grainy), VL-7B int4; small top-level projections + fp32 VAE
                //     stay full precision. Load PEAK is still ~41 GB (bf16 DiT load);
                //     pre-quantized weight hosting to cut peak toward ~16 GB resident+peak
                //     is the tracked follow-up. 4-step DMD.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 57_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 22_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                IEditContract.descriptor(
                    name: "qwen-image-edit-turbo",
                    summary: "Qwen-Image-Edit-2511 fast tier: Lightning 4-step DMD LoRA "
                        + "applied at runtime over the 20B DiT. Multi-image fusion, "
                        + "identity-preserving edits, 1024²-area output, 4-step single-pass.",
                    modes: [turboMode]
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
        let snapshot = URL(fileURLWithPath: configuration.snapshotPath)
        guard FileManager.default.fileExists(
            atPath: snapshot.appendingPathComponent("transformer").path)
        else { throw QwenImageEditTurboPackageError.unreadableSnapshot(snapshot.path) }
        let lora = URL(fileURLWithPath: configuration.loraPath)
        guard FileManager.default.fileExists(atPath: lora.path)
        else { throw QwenImageEditTurboPackageError.unreadableLoRA(lora.path) }

        // DiT first so its bf16 originals can free before the encoder loads (caps peak).
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        // Optional int4 tier: quantize the bulk attn + feed-forward Linears (the modulation
        // linears are left full precision — they're small and quality-sensitive). Must run
        // BEFORE apply(): quantized layers are then wrapped as QLoRALinear so the LoRA rides
        // the quantized base. eval() materializes int4 and frees the bf16 weights now.
        if let bits = configuration.ditBits {
            let modBits = configuration.modulationBits
            // Per-layer bits: attn + feed-forward at ditBits; modulation at modulationBits
            // (higher precision, it's conditioning-critical) if requested; skip the rest.
            quantize(model: transformer) { path, module
                -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                guard module is Linear, path.contains("transformer_blocks") else { return nil }
                if path.contains(".attn.") || path.contains("_mlp.") {
                    return (groupSize: 64, bits: bits, mode: .affine)
                }
                if let mb = modBits, path.contains(".img_mod") || path.contains(".txt_mod") {
                    return (groupSize: 64, bits: mb, mode: .affine)
                }
                return nil
            }
            eval(transformer)
        }
        // Runtime DMD: apply Lightning in the activation path (survives bf16/int4; a fused
        // bf16 snapshot would lose it).
        try QwenImageEditLoRA.apply(
            diffusersLoRA: lora, to: transformer, strength: configuration.strength)

        let encoder = try await QwenVLPromptEncoder.load(
            snapshot: snapshot, bits: configuration.encoderBits)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        generator = QwenImageEditGenerator(
            encoder: encoder, transformer: transformer, vae: vae)
    }

    public func unload() async {
        generator = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .imageEdit, let edit = request as? IEditRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        guard !edit.images.isEmpty else { throw QwenImageEditTurboPackageError.imageDecode }
        try Task.checkCancellation()
        let inputs = try edit.images.map { try Self.decodeRGB($0.data) }

        let (pixels, w, h) = try generator.generate(
            images: inputs,
            prompt: edit.prompt,
            negativePrompt: edit.negativePrompt ?? " ",
            steps: edit.steps ?? configuration.defaultSteps,
            trueCFGScale: edit.guidanceScale.map(Float.init)
                ?? configuration.defaultTrueCFGScale,
            seed: edit.seed ?? 0,
            progress: { _, _ in })

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
        else { throw QwenImageEditTurboPackageError.imageDecode }
        let w = cg.width
        let h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw QwenImageEditTurboPackageError.imageDecode }
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
        else { throw QwenImageEditTurboPackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw QwenImageEditTurboPackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImageEditTurboPackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw QwenImageEditTurboPackageError.pngEncode
        }
        return out as Data
    }
}

extension QwenImageEditTurboPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(QwenImageEditTurboPackage.self)
    }
}
