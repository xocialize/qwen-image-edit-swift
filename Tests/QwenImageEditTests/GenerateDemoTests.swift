// S4 e2e: full Swift edit render (eye gate). Saves ~/Desktop/qwen-edit-swift-demo.png.
//
// Run: QIE_DEMO=1 swift test --filter GenerateDemoTests

import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers
import XCTest

@testable import QwenImageEdit

final class GenerateDemoTests: XCTestCase {
    func testEditDemo() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_DEMO"] == "1", "QIE_DEMO=1")

        let modelDir = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")
        let goldens = URL(
            fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: goldens.appendingPathComponent("goldens_meta.json")))
            as! [String: Any]
        let image = try EncoderParityTests.loadRGB(
            url: URL(fileURLWithPath: meta["input_image"] as! String))

        let encoder = try await QwenVLPromptEncoder.load(snapshot: modelDir)
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: modelDir.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: modelDir.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImageEditGenerator(
            encoder: encoder, transformer: transformer, vae: vae)

        let start = Date()
        let (pixels, w, h) = try await generator.generate(
            image: image,
            prompt: meta["prompt"] as! String,
            negativePrompt: meta["negative_prompt"] as! String,
            steps: 20,
            trueCFGScale: 4.0,
            seed: 42,
            progress: { step, total in print("step \(step)/\(total)", Date()) })
        print("generated \(w)x\(h) in \(Date().timeIntervalSince(start))s")

        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qwen-edit-swift-demo.png")
        try Self.writePNG(pixels: pixels, width: w, height: h, to: out)
        print("saved \(out.path)")
    }

    static func writePNG(pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw QwenImageEditError.invalidInput("CGContext") }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithData(
                  NSMutableData() as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImageEditError.invalidInput("PNG encode") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw QwenImageEditError.invalidInput("PNG finalize")
        }
        // CGImageDestinationCreateWithData wrote into its own data; re-create with URL
        // for simplicity:
        guard let urlDest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImageEditError.invalidInput("PNG dest") }
        CGImageDestinationAddImage(urlDest, image, nil)
        guard CGImageDestinationFinalize(urlDest) else {
            throw QwenImageEditError.invalidInput("PNG write")
        }
    }
}
