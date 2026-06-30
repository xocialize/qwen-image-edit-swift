// TeleStyleV2 e2e: content+style transfer on the fused (style + Lightning-4step DMD)
// Qwen-Image-Edit-2511 transformer. Eye gate — saves PNGs to ~/Desktop.
//
// Run: TELESTYLE_DEMO=1 swift test --filter TeleStyleDemoTests
//
// Model dir must be the merged TeleStyleV2-2511 snapshot (merged transformer/ + the
// base vae/ text_encoder/ processor/). Sample content/style pairs come from the repo's
// qwenstyleref/ (Tele-AI/TeleStyleV2 GitHub).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers
import XCTest

@testable import QwenImageEdit

final class TeleStyleDemoTests: XCTestCase {
    static let modelDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/TeleStyleV2-2511")
    static let assets = URL(
        fileURLWithPath: "/Users/dustinnielson/Development/telestyle-work/tele_assets")
    static let stylePrompt =
        "Style Transfer the style of Figure 2 to Figure 1, and keep the content and "
        + "characteristics of Figure 1."

    func testStyleTransfer() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["TELESTYLE_DEMO"] == "1", "TELESTYLE_DEMO=1")

        let content = try EncoderParityTests.loadRGB(
            url: Self.assets.appendingPathComponent("content_1.webp"))
        let style = try EncoderParityTests.loadRGB(
            url: Self.assets.appendingPathComponent("style_1.jpg"))
        print("content \(content.width)x\(content.height)  style \(style.width)x\(style.height)")

        let encoder = try await QwenVLPromptEncoder.load(snapshot: Self.modelDir)
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: Self.modelDir.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: Self.modelDir.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImageEditGenerator(
            encoder: encoder, transformer: transformer, vae: vae)

        let start = Date()
        let (pixels, w, h) = try await generator.generate(
            images: [content, style],           // image 1 = content, image 2 = style
            prompt: Self.stylePrompt,
            steps: 4,                            // Lightning/DMD 4-step
            trueCFGScale: 1.0,                   // DMD: no CFG (single positive pass)
            seed: 123,
            progress: { step, total in print("step \(step)/\(total)", Date()) })
        print("style-transfer \(w)x\(h) in \(Date().timeIntervalSince(start))s")

        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/telestyle-content1-style1.png")
        try GenerateDemoTests.writePNG(pixels: pixels, width: w, height: h, to: out)
        print("saved \(out.path)")
    }

    // Single-image edit at 4 steps — isolates "merge loads + Lightning works" from the
    // multi-image path. Uses the content image alone with a plain edit prompt.
    func testSingleImage4Step() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["TELESTYLE_DEMO"] == "1", "TELESTYLE_DEMO=1")

        let content = try EncoderParityTests.loadRGB(
            url: Self.assets.appendingPathComponent("content_1.webp"))
        let encoder = try await QwenVLPromptEncoder.load(snapshot: Self.modelDir)
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: Self.modelDir.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: Self.modelDir.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImageEditGenerator(
            encoder: encoder, transformer: transformer, vae: vae)

        let (pixels, w, h) = try await generator.generate(
            image: content,
            prompt: "make it a watercolor painting",
            steps: 4, trueCFGScale: 1.0, seed: 123,
            progress: { step, total in print("step \(step)/\(total)") })
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/telestyle-single-4step.png")
        try GenerateDemoTests.writePNG(pixels: pixels, width: w, height: h, to: out)
        print("saved \(out.path)")
    }
}
