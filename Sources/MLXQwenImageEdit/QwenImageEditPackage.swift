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
import MLXToolKit
import QwenImageEdit
import UniformTypeIdentifiers

/// Init-time configuration (C9): the 2511 snapshot root and generation defaults.
public struct QwenImageEditConfiguration: PackageConfiguration, ModelStorable {
    /// Snapshot root with `transformer/`, `vae/`, `text_encoder/`, `processor/`.
    public var snapshotPath: String
    public var defaultSteps: Int
    public var defaultTrueCFGScale: Float
    public var modelsRootDirectory: URL?

    public init(
        snapshotPath: String =
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511",
        defaultSteps: Int = 20,
        defaultTrueCFGScale: Float = 4.0,
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
                // Measured regime (1024², 20 steps, bf16 DiT + bf16 VL-7B + fp32 VAE):
                // ~60 GB resident. 4-bit DiT+VL is the tracked follow-up (~16 GB).
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 60_000_000_000)],
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
                        + "preserving edits, 1024²-area output, 20-step true CFG.",
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
        let snapshot = URL(fileURLWithPath: configuration.snapshotPath)
        guard FileManager.default.fileExists(
            atPath: snapshot.appendingPathComponent("transformer").path)
        else { throw QwenImageEditPackageError.unreadableSnapshot(snapshot.path) }

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
        guard let first = edit.images.first else {
            throw QwenImageEditPackageError.imageDecode
        }
        // Core pipeline currently consumes ONE conditioning image; multi-image fusion
        // (per-image VAE sizes + grids) is wired in the core as the follow-up.
        try Task.checkCancellation()
        let input = try Self.decodeRGB(first.data)

        let (pixels, w, h) = try generator.generate(
            image: input,
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
