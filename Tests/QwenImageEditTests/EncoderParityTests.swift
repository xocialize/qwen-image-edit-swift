// S3 gate: VL-7B prompt encoding vs the P2 golden (enc_prompt.safetensors).
//
// The golden was captured with diffusers fp32 CPU on the fox image; we run bf16 GPU
// (production regime). Gate: token count exact + cosine per the bf16 calibration.
//
// Run: QIE_PARITY=1 swift test --filter EncoderParityTests

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import QwenImageEdit

final class EncoderParityTests: XCTestCase {
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
    static let modelDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")

    /// Decode a PNG to interleaved RGB8 in sRGB (what PIL convert("RGB") sees).
    static func loadRGB(url: URL) throws -> (rgb: [UInt8], width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw QwenImageEditError.invalidInput("unreadable image \(url.path)") }
        let w = cg.width
        let h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw QwenImageEditError.invalidInput("CGContext failed") }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return (rgb, w, h)
    }

    func testEncodePromptGolden() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PARITY"] == "1",
            "set QIE_PARITY=1 to run (loads the 7B VL encoder)")

        let enc = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("enc_prompt.safetensors"))
        let metaData = try Data(
            contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let prompt = meta["prompt"] as! String
        let negative = meta["negative_prompt"] as! String
        let imagePath = meta["input_image"] as! String

        let encoder = try await QwenVLPromptEncoder.load(snapshot: Self.modelDir)
        let image = try Self.loadRGB(url: URL(fileURLWithPath: imagePath))

        func gate(_ text: String, _ goldenKey: String) throws -> Float {
            let ours = try encoder.encode(prompt: text, images: [image])
            let ref = enc[goldenKey]!
            XCTAssertEqual(
                ours.dim(1), ref.dim(1),
                "\(goldenKey): token count \(ours.dim(1)) != golden \(ref.dim(1))")
            let a = ours.asType(.float32).flattened()
            let b = ref.asType(.float32).flattened()
            let cos = sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)) + 1e-12)
            eval(cos)
            let c = cos.item(Float.self)
            print("\(goldenKey): cosine \(c) (S=\(ours.dim(1)))")
            return c
        }

        let cosPos = try gate(prompt, "prompt_embeds")
        XCTAssertGreaterThanOrEqual(cosPos, 0.995)
        let cosNeg = try gate(negative, "neg_embeds")
        XCTAssertGreaterThanOrEqual(cosNeg, 0.995)
    }
}
