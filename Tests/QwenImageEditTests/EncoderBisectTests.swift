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

        // --- stage e: position ids + merged rope cos/sin vs HF rope_golden ---
        let rope = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("rope_golden.safetensors"))
        let refPos = rope["position_ids"]!.asType(.int32)  // (3, 1, 291)
        let fullIds = try Self.buildFullIds(encoder: encoder, prompt: prompt, padCount: 196)
        let ourPos = QwenVLPromptEncoder.positionIds(
            ids: fullIds, frames: [frame], mergeSize: 2, imagePadId: encoder.imagePadId)
        XCTAssertEqual(ourPos.shape, refPos.shape, "position shape")
        let posDiff = max(abs(ourPos.asType(.int32) - refPos)).item(Int32.self)
        print("stage e positions: max_abs \(posDiff)")

        let (cosT, sinT) = MRoPE.cosSin(
            positionIds: ourPos, headDim: 128, theta: 1_000_000, mropeSection: [16, 24, 24])
        // ours: (B,1,T,128); golden merged: (1,T,128)
        let cosDiff = max(abs(cosT.asType(.float32).squeezed(axis: 1) - rope["cos_merged"]!))
            .item(Float.self)
        let sinDiff = max(abs(sinT.asType(.float32).squeezed(axis: 1) - rope["sin_merged"]!))
            .item(Float.self)
        print("stage e rope: cos max_abs \(cosDiff)  sin max_abs \(sinDiff)")

        // --- stage f: per-layer ladder vs HF hidden_states (find divergence layer) ---
        let layers = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder_layers.safetensors"))
        func visionCos(_ a: MLXArray, _ ref: MLXArray) -> Float {
            // vision span in the FULL 291-seq: 64 (system) + 5 (Picture 1: +start) ... use 69..<265
            let x = a[0..., 69..<265, 0...].asType(.float32).flattened()
            let y = ref[0..., 69..<265, 0...].asType(.float32).flattened()
            return (sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)) + 1e-12)).item(Float.self)
        }
        // layer_0 = post-merge embeddings
        print("stage f layer_0 (embeds) vision cosine: \(visionCos(embeds, layers["layer_0"]!))")
        var h = embeds
        var checkpoints: [Int: MLXArray] = [:]
        for (i, layer) in encoder.model.layers.enumerated() {
            h = layer(h, positionIds: positionIds, mask: .causal, cache: nil)
            checkpoints[i + 1] = h
        }
        for idx in [1, 2, 7, 14, 21, 27] {
            if let ref = layers["layer_\(idx)"], let ours = checkpoints[idx] {
                print("stage f layer_\(idx) vision cosine: \(visionCos(ours, ref))")
            }
        }

        // --- stage g: layer-0 full-seq ids check + layer-1 internal ops ---
        let l0ref = layers["layer_0"]!
        let embedsMaxAbs = max(abs(embeds.asType(.float32) - l0ref)).item(Float.self)
        print("stage g embeds full-seq max_abs: \(embedsMaxAbs)")

        let internals = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder_layer1.safetensors"))
        let layer0 = encoder.model.layers[0]
        let ln1 = layer0.inputNorm(embeds)
        print("stage g ln1 vision cosine: \(visionCos(ln1, internals["ln1_out"]!))")
        // attention over GOLDEN ln1 input — isolates the attention op itself
        let attnOut = layer0.attention(
            internals["ln1_out"]!.asType(embeds.dtype), positionIds: positionIds,
            mask: .causal, cache: nil)
        print("stage g attn(golden ln1) vision cosine: \(visionCos(attnOut, internals["attn_out"]!))")
        let attnFull = attnOut.asType(.float32).flattened()
        let attnRefFull = internals["attn_out"]!.flattened()
        let attnCosFull = (sum(attnFull * attnRefFull)
            / (sqrt(sum(attnFull * attnFull)) * sqrt(sum(attnRefFull * attnRefFull)) + 1e-12))
            .item(Float.self)
        print("stage g attn(golden ln1) FULL cosine: \(attnCosFull)")
        // MLP over golden ln2 input
        let mlpOut = layer0.mlp(internals["ln2_out"]!.asType(embeds.dtype))
        print("stage g mlp(golden ln2) vision cosine: \(visionCos(mlpOut, internals["mlp_out"]!))")

        // --- stage h: attention sub-ops vs HF attn0 internals ---
        let attn0 = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder_attn0.safetensors"))
        let attnMod = layer0.attention
        let x = internals["ln1_out"]!.asType(embeds.dtype)
        func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
            max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
        }
        let qp = attnMod.wq(x)
        let kp = attnMod.wk(x)
        let vp = attnMod.wv(x)
        print("stage h q_proj max_abs: \(maxAbs(qp, attn0["q_proj"]!))")
        print("stage h k_proj max_abs: \(maxAbs(kp, attn0["k_proj"]!))")
        print("stage h v_proj max_abs: \(maxAbs(vp, attn0["v_proj"]!))")

        let B = x.dim(0); let L = x.dim(1)
        var q4 = qp.reshaped(B, L, 28, 128).transposed(0, 2, 1, 3)
        var k4 = kp.reshaped(B, L, 4, 128).transposed(0, 2, 1, 3)
        let v4 = vp.reshaped(B, L, 4, 128).transposed(0, 2, 1, 3)
        let (cosA, sinA) = MRoPE.cosSin(
            positionIds: ourPos, headDim: 128, theta: 1_000_000, mropeSection: [16, 24, 24])
        (q4, k4) = MRoPE.apply(q: q4, k: k4, cos: cosA.asType(q4.dtype), sin: sinA.asType(q4.dtype))
        print("stage h q_rot max_abs: \(maxAbs(q4, attn0["q_rot"]!))")
        print("stage h k_rot max_abs: \(maxAbs(k4, attn0["k_rot"]!))")

        let ctx = MLXFast.scaledDotProductAttention(
            queries: q4, keys: k4, values: v4, scale: 1.0 / Float(128).squareRoot(),
            mask: .causal)
        let ctxFlat = ctx.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        print("stage h sdpa(ctx) max_abs vs o_in: \(maxAbs(ctxFlat, attn0["o_in"]!))")
        let cA = ctxFlat.asType(.float32).flattened()
        let cB = attn0["o_in"]!.flattened()
        let ctxCos = (sum(cA * cB) / (sqrt(sum(cA * cA)) * sqrt(sum(cB * cB)) + 1e-12))
            .item(Float.self)
        print("stage h sdpa(ctx) FULL cosine: \(ctxCos)")

        // --- stage i: GQA-vs-mask discriminator ---
        // (1) SDPA with k/v explicitly repeated to 28 heads (kills GQA broadcasting)
        let kRep = repeated(k4, count: 7, axis: 1)
        let vRep = repeated(v4, count: 7, axis: 1)
        let ctx1 = MLXFast.scaledDotProductAttention(
            queries: q4, keys: kRep, values: vRep, scale: 1.0 / Float(128).squareRoot(),
            mask: .causal)
        let ctx1F = ctx1.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        print("stage i sdpa(k/v repeated x7) max_abs: \(maxAbs(ctx1F, attn0["o_in"]!))")

        // (2) manual attention: explicit matmul + additive causal mask + softmax
        let scores = matmul(q4, kRep.transposed(0, 1, 3, 2)) * (1.0 / Float(128).squareRoot())
        let iota = MLXArray(0..<L)
        let causal = MLX.which(
            iota[0..., .newAxis] .>= iota[.newAxis, 0...], MLXArray(Float(0)),
            MLXArray(-Float.infinity))
        let probs = softmax(scores + causal[.newAxis, .newAxis, 0..., 0...], axis: -1)
        let ctx2 = matmul(probs, vRep)
        let ctx2F = ctx2.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        print("stage i manual attention max_abs: \(maxAbs(ctx2F, attn0["o_in"]!))")
    }

    static func buildFullIds(encoder: QwenVLPromptEncoder, prompt: String, padCount: Int)
        throws -> [Int]
    {
        try buildIds(encoder: encoder, prompt: prompt, padCount: padCount)
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
