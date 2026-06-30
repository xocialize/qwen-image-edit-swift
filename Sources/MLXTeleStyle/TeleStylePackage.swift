// MLXEngine package for TeleStyleV2 — content-preserving image style transfer.
//
// TeleStyleV2 (Apache-2.0, Tele-AI) is the QwenImageEdit-2511 base with two LoRAs at
// strength 1.0: the TeleStyleV2 style LoRA + the QIE-2511 Lightning 4-step (DMD) LoRA.
// Both are applied at RUNTIME (QwenImageEditLoRA.apply, rank-stacked) over the base
// transformer — NOT pre-fused. Fusing into a bf16 snapshot rounds the DMD LoRA away
// (its per-weight deltas sit below bf16 ULP), which silently broke the 4-step behavior;
// runtime application adds each adapter's low-rank term in the activation path, where it
// survives bf16. (This retires the external Python merge + the fused TeleStyleV2-2511
// snapshot.) Same core (`QwenImageEdit`), no architecture change, no role token: image 0
// is the content, image 1 the style reference; the LoRAs supply the behavior.
//
// Surface: the canonical `imageEdit` capability with a `styleTransfer` mode tag
// (per C4 — one multi-image editor, the tag declares intent; not a new capability).
// Defaults: 4 steps, true-CFG 1.0 (DMD: single positive pass).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXToolKit
import QwenImageEdit
import UniformTypeIdentifiers

/// The `styleTransfer` imageEdit mode (rawValue matches `MLXToolKit.Mode.styleTransfer`
/// in mlx-engine-swift ≥ next tag; used as a literal here for the current pinned tag).
public let styleTransferMode = Mode(rawValue: "styleTransfer")

/// Init-time configuration (C9): the BASE 2511 snapshot + the two LoRAs + DMD defaults.
public struct TeleStyleConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Base 2511 snapshot root with `transformer/`, `vae/`, `text_encoder/`, `processor/`.
    public var basePath: String
    /// Diffusers-format TeleStyleV2 style LoRA (.safetensors).
    public var styleLoRAPath: String
    /// Diffusers-format Lightning 4-step DMD LoRA (.safetensors).
    public var lightningLoRAPath: String
    /// Per-adapter strength multipliers (1.0 = the merge's `set_adapters(..., 1.0)`).
    public var styleStrength: Float
    public var lightningStrength: Float
    public var defaultSteps: Int
    public var defaultTrueCFGScale: Float
    public var modelsRootDirectory: URL?

    /// Always bf16 (same base weights as QwenImageEdit; the LoRAs add only rank factors).
    public var quant: Quant { .bf16 }

    public init(
        basePath: String =
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511",
        styleLoRAPath: String =
            "/Users/dustinnielson/Development/telestyle-work/loras/"
            + "diffusers-TeleStyleV2-QIE-2511-Lora-bf16.safetensors",
        lightningLoRAPath: String =
            "/Users/dustinnielson/Development/telestyle-work/loras/"
            + "QIE-2511-Lightning-4steps-V1.0-bf16.safetensors",
        styleStrength: Float = 1.0,
        // Strength eval (content_1/style_1, 4 steps): the Lightning DMD is UNDER-applied at
        // the diffusers default (1.0 = 0.125 effective) — output stays soft, like style-
        // only. At ~4 (0.5 effective) the DMD fully engages and 4-step output is crisp
        // flat-illustration (matching the 16-step@1.0 reference, at a quarter the steps).
        // So strength — not step count — was the real lever; default 4.0 + 4 steps.
        lightningStrength: Float = 4.0,
        defaultSteps: Int = 4,
        defaultTrueCFGScale: Float = 1.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.basePath = basePath
        self.styleLoRAPath = styleLoRAPath
        self.lightningLoRAPath = lightningLoRAPath
        self.styleStrength = styleStrength
        self.lightningStrength = lightningStrength
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case basePath, styleLoRAPath, lightningLoRAPath
        case styleStrength, lightningStrength, defaultSteps, defaultTrueCFGScale
    }
}

public enum TeleStylePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case unreadableLoRA(String)
    case imageDecode
    case pngEncode
    case noImages

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "TeleStyleV2 base snapshot not readable at \(p)."
        case .unreadableLoRA(let p): return "TeleStyleV2 LoRA not readable at \(p)."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        case .noImages: return "imageEdit/styleTransfer requires at least one image."
        }
    }
}

@InferenceActor
public final class TeleStylePackage: ModelPackage {
    public typealias Configuration = TeleStyleConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "Tele-AI/TeleStyleV2", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Same split as the QwenImageEdit base — same bf16 weights; the runtime LoRAs add
                // only small rank factors to the resident DiT. Per-stage eviction (P2): the VL-7B
                // encoder is a per-request transient, not a resident. Measured M5 Max, 1024²/8-step
                // (QIE_MEMBENCH in MLXQwenImageEditTests): resident floor (DiT 40.9 GB + fp32 VAE
                // 0.5 GB) = 41.4 GB; activation (peak − floor, ~16.6 GB transient encoder) ≈ 17.9 GB
                // → 21 GB at +20% headroom. Old flat 60 GB folded the encoder into resident. DMD
                // tier runs 4 steps; 4-bit is the Turbo int4 tier.
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
                    name: "telestyle-v2",
                    summary: "TeleStyleV2 content-preserving style transfer (Qwen-Image-"
                        + "Edit-2511 + fused style/DMD LoRAs): image 0 = content, image 1 "
                        + "= style; 4-step DMD, 1024²-area output. Also serves plain "
                        + "instruction edits (content image only).",
                    modes: [styleTransferMode]
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
        let base = URL(fileURLWithPath: configuration.basePath)
        guard FileManager.default.fileExists(
            atPath: base.appendingPathComponent("transformer").path)
        else { throw TeleStylePackageError.unreadableSnapshot(base.path) }
        let style = URL(fileURLWithPath: configuration.styleLoRAPath)
        let lightning = URL(fileURLWithPath: configuration.lightningLoRAPath)
        for lora in [style, lightning] where !FileManager.default.fileExists(atPath: lora.path) {
            throw TeleStylePackageError.unreadableLoRA(lora.path)
        }

        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: base.appendingPathComponent("transformer"), dtype: .bfloat16)
        // Runtime, rank-stacked: style + Lightning together in the activation path (a bf16
        // fuse would lose the DMD adapter).
        try QwenImageEditLoRA.apply(
            diffusersLoRAs: [
                (style, configuration.styleStrength),
                (lightning, configuration.lightningStrength),
            ],
            to: transformer)
        // The DiT (with the fused-runtime LoRAs) + VAE stay resident; the VL-7B encoder is
        // loaded per request and evicted before the denoise peak (efficiency contract 1.14.0).
        let vae = try QwenImageEditWeights.loadVAE(
            directory: base.appendingPathComponent("vae"), dtype: .float32)
        generator = QwenImageEditGenerator(
            encoderProvider: { try await QwenVLPromptEncoder.load(snapshot: base) },
            transformer: transformer, vae: vae)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .imageEdit, let edit = request as? IEditRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        guard !edit.images.isEmpty else { throw TeleStylePackageError.noImages }
        try Task.checkCancellation()

        // Decode every conditioning image in prompt order (image 0 = content, image 1 =
        // style for styleTransfer); the core packs per-image VAE latents + grids.
        let inputs = try edit.images.map { try Self.decodeRGB($0.data) }

        let (pixels, w, h) = try await generator.generate(
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

    /// PNG/JPEG/WebP Data -> interleaved RGB8 in sRGB.
    nonisolated static func decodeRGB(_ data: Data) throws
        -> (rgb: [UInt8], width: Int, height: Int)
    {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw TeleStylePackageError.imageDecode }
        let w = cg.width
        let h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw TeleStylePackageError.imageDecode }
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
        else { throw TeleStylePackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw TeleStylePackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw TeleStylePackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw TeleStylePackageError.pngEncode
        }
        return out as Data
    }
}

extension TeleStylePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(TeleStylePackage.self)
    }
}
