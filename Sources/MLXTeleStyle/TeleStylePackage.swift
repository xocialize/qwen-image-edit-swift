// MLXEngine package for TeleStyleV2 — content-preserving image style transfer.
//
// TeleStyleV2 (Apache-2.0, Tele-AI) is the QwenImageEdit-2511 base with two LoRAs
// fused at scale 1.0: the TeleStyleV2 style LoRA + the QIE-2511 Lightning 4-step
// (DMD) LoRA. It is therefore the SAME core (`QwenImageEdit`) over a different
// (fused) snapshot — there is no architecture change, no role token: image 0 is the
// content, image 1 the style reference, and the fused LoRA supplies the behavior.
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

/// Init-time configuration (C9): the fused TeleStyleV2-2511 snapshot + DMD defaults.
public struct TeleStyleConfiguration: PackageConfiguration, ModelStorable {
    /// Snapshot root with the FUSED `transformer/` + base `vae/`, `text_encoder/`,
    /// `processor/` (the merge tool produces this layout).
    public var snapshotPath: String
    public var defaultSteps: Int
    public var defaultTrueCFGScale: Float
    public var modelsRootDirectory: URL?

    public init(
        snapshotPath: String =
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/TeleStyleV2-2511",
        defaultSteps: Int = 4,
        defaultTrueCFGScale: Float = 1.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, defaultSteps, defaultTrueCFGScale
    }
}

public enum TeleStylePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode
    case noImages

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "TeleStyleV2 snapshot not readable at \(p)."
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
                // Same footprint as the QwenImageEdit base (fused LoRA adds no params):
                // ~60 GB resident bf16 (20B DiT + bf16 VL-7B + fp32 VAE). 4-bit is the
                // tracked follow-up (~16 GB). DMD tier runs 4 steps.
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 60_000_000_000)],
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
        let snapshot = URL(fileURLWithPath: configuration.snapshotPath)
        guard FileManager.default.fileExists(
            atPath: snapshot.appendingPathComponent("transformer").path)
        else { throw TeleStylePackageError.unreadableSnapshot(snapshot.path) }

        let encoder = try await QwenVLPromptEncoder.load(snapshot: snapshot)
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
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
        guard !edit.images.isEmpty else { throw TeleStylePackageError.noImages }
        try Task.checkCancellation()

        // Decode every conditioning image in prompt order (image 0 = content, image 1 =
        // style for styleTransfer); the core packs per-image VAE latents + grids.
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
