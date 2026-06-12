// S3 bisect: which stage diverges — preprocessing pixels, ViT, or the LLM forward?
// Uses encoder_stages.safetensors (PT fp32 dumps on the SAME lanczos-resized image,
// saved as cond_resized.png — isolating our Lanczos out of stages a/b).
//
// Run: QIE_PARITY=1 swift test --filter EncoderBisectTests

import CoreGraphics
import Foundation
import ImageIO
import MLX
import Qwen25VL
import XCTest

@testable import QwenImageEdit

final class EncoderBisectTests: XCTestCase {
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
    static let modelDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")

    func testStages() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PARITY"] == "1", "QIE_PARITY=1")
        let fp32CPU = ProcessInfo.processInfo.environment["QIE_FP32_CPU"] == "1"
        if fp32CPU { Device.setDefault(device: Device(.cpu)) }

        let stages = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder_stages.safetensors"))
        let condImage = try EncoderParityTests.loadRGB(
            url: Self.goldens.appendingPathComponent("cond_resized.png"))

        // --- stage a: preprocessing (bicubic smart-resize + normalize + patchify) ---
        let (th, tw) = try Qwen25VLImageProcessing.targetSize(
            height: condImage.height, width: condImage.width,
            factor: Qwen25VLImageProcessing.patchSize * Qwen25VLImageProcessing.mergeSize,
            minPixels: 3136, maxPixels: 12_845_056)
        let bicubic = PILResize.resize(
            rgb: condImage.rgb, width: condImage.width, height: condImage.height,
            outWidth: tw, outHeight: th)
        var chw = [Float](repeating: 0, count: 3 * th * tw)
        let plane = th * tw
        let mean = Qwen25VLImageProcessing.imageMean
        let std = Qwen25VLImageProcessing.imageStd
        for i in 0..<plane {
            let p = i * 3
            let r: Float = Float(bicubic[p]) / 255
            let g: Float = Float(bicubic[p + 1]) / 255
            let b: Float = Float(bicubic[p + 2]) / 255
            chw[i] = (r - mean[0]) / std[0]
            chw[plane + i] = (g - mean[1]) / std[1]
            chw[2 * plane + i] = (b - mean[2]) / std[2]
        }
        let (patches, frame) = try Qwen25VLImageProcessing.patchify(
            images: [MLXArray(chw, [1, 3, th, tw])])
        let refPixels = stages["pixel_values"]!
        XCTAssertEqual(patches.shape, refPixels.shape, "pixel_values shape")
        let pixMaxAbs = max(abs(patches - refPixels)).item(Float.self)
        print("stage a (pixels): max_abs \(pixMaxAbs)  grid \(frame)")

        // --- stage b: ViT (theirs-pixels -> our ViT) vs visual_features (pre-merger
        //     in this dump: 784x1280; ours returns merged 196x3584 — compare only if
        //     shapes align, else report) ---
        let encoder = try await QwenVLPromptEncoder.load(
            snapshot: Self.modelDir, dtype: fp32CPU ? .float32 : .bfloat16)
        let visionDtype: DType = fp32CPU ? .float32 : .bfloat16
        let ourFeatures = encoder.vision(
            refPixels.asType(visionDtype), frames: [frame])
        let refVisual = stages["visual_merged"] ?? stages["visual_features"]!
        if ourFeatures.shape == refVisual.shape {
            let a = ourFeatures.asType(.float32).flattened()
            let b = refVisual.flattened()
            let cos = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)) + 1e-12))
                .item(Float.self)
            print("stage b (ViT): cosine \(cos)")
        } else {
            print(
                "stage b (ViT): shape ours \(ourFeatures.shape) vs ref \(refVisual.shape) — pre/post-merger mismatch, skipping direct compare"
            )
        }

        // --- stage c: full forward vs hidden_last (their pixels via our whole path
        //     happens inside encode(); here compare the final hidden over all 291) ---
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json")))
            as! [String: Any]
        let prompt = meta["prompt"] as! String
        let fox = try EncoderParityTests.loadRGB(
            url: URL(fileURLWithPath: meta["input_image"] as! String))
        let ours = try encoder.encode(prompt: prompt, images: [fox])
        let refHidden = stages["hidden_last"]![0..., QwenVLPromptEncoder.dropIdx..., 0...]
        XCTAssertEqual(ours.dim(1), refHidden.dim(1))
        // Per-token-range cosines: text prefix (before vision), vision span, tail.
        func cosRange(_ lo: Int, _ hi: Int, _ label: String) {
            let a = ours[0..., lo..<hi, 0...].asType(.float32).flattened()
            let b = refHidden[0..., lo..<hi, 0...].flattened()
            let c = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)) + 1e-12))
                .item(Float.self)
            print("stage c hidden[\(label) \(lo)..<\(hi)]: cosine \(c)")
        }
        let S = ours.dim(1)
        // After drop 64: "Picture 1: " ≈ tokens 0..5, vision 5..201, text 201..S.
        cosRange(0, 5, "pre-vision")
        cosRange(5, 201, "vision-span")
        cosRange(201, S, "tail")
        cosRange(0, S, "all")

        // --- stage d: golden fp32 merged features through OUR LLM path ---
        // If this snaps to ~1, the LLM integration (mask/positions/merge) is right
        // and the 0.83 is bf16 ViT-feature error amplified by massive activations.
        let goldenFeatures = stages["visual_merged"]!
        let ids = try Self.buildIds(encoder: encoder, prompt: prompt, padCount: 196)
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)
        var embeds = encoder.model.embedTokens(inputIds)
        let padPositions = ids.enumerated().filter { $0.element == encoder.imagePadId }
            .map(\.offset)
        embeds = try QwenVLPromptEncoder.mergeImageFeatures(
            textEmbeds: embeds, imageFeatures: goldenFeatures.asType(embeds.dtype),
            padPositions: padPositions)
        let positionIds = QwenVLPromptEncoder.positionIds(
            ids: ids, frames: [frame], mergeSize: 2, imagePadId: encoder.imagePadId)
        let hidden = encoder.model(
            inputEmbeddings: embeds, positionIds: positionIds, mask: .causal, caches: nil)
        let oursD = hidden[0..., QwenVLPromptEncoder.dropIdx..., 0...]
        let aD = oursD.asType(.float32).flattened()
        let bD = refHidden.flattened()
        let cD = (sum(aD * bD) / (sqrt(sum(aD * aD)) * sqrt(sum(bD * bD)) + 1e-12))
            .item(Float.self)
        print("stage d (golden features -> our LLM): cosine \(cD)")
    }

    static func buildIds(encoder: QwenVLPromptEncoder, prompt: String, padCount: Int)
        throws -> [Int]
    {
        let text = QwenVLPromptEncoder.promptTemplatePrefix
            + "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + prompt
            + QwenVLPromptEncoder.promptTemplateSuffix
        var ids = encoder.tokenizer.encode(text: text, addSpecialTokens: false)
        guard let idx = ids.firstIndex(of: encoder.imagePadId) else {
            throw QwenImageEditError.invalidInput("no pad")
        }
        ids.replaceSubrange(idx...idx, with: Array(repeating: encoder.imagePadId, count: padCount))
        return ids
    }
}
