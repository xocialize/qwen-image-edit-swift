// Qwen2.5-VL-7B prompt encoder for Qwen-Image-Edit-2511 — the diffusers
// QwenImageEditPlusPipeline.encode_prompt path.
//
// Backbone + ViT + HF-exact preprocessing come from qwen25vl-mlx-swift
// (parity-locked vs PT). This file adds the edit-plus specifics:
//   - the 2511 snapshot's HF prefixes (`model.*` / `visual.*` vs the
//     mlx-community `language_model.model.*` / `vision_tower.*`)
//   - the edit-plus chat template with "Picture {n}: " image prefixes and
//     drop_idx 64 (the tokenized system block)
//   - the conditioning resize chain: diffusers VaeImageProcessor LANCZOS to the
//     384²-area /32-rounded size, THEN the HF processor's BICUBIC smart-resize
//   - last-hidden-state extraction (final norm applied = HF hidden_states[-1])
//
// lm_head is untied on 7B but never used for feature extraction — skipped at load.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Qwen25VL
import Tokenizers

public final class QwenVLPromptEncoder {
    public let model: Qwen25VLModel
    public let vision: QVLVision.VisionModel
    public let tokenizer: any Tokenizers.Tokenizer
    let imagePadId: Int
    let minPixels: Int
    let maxPixels: Int
    let dtype: DType

    /// pipeline_qwenimage_edit_plus.py L215-217 — tokenizes to exactly 64 tokens.
    static let promptTemplatePrefix =
        "<|im_start|>system\n"
        + "Describe the key features of the input image (color, shape, size, texture, "
        + "objects, background), then explain how the user's text instruction should alter "
        + "or modify the image. Generate a new image that meets the user's requirements "
        + "while maintaining consistency with the original input where appropriate."
        + "<|im_end|>\n<|im_start|>user\n"
    static let promptTemplateSuffix = "<|im_end|>\n<|im_start|>assistant\n"
    static let dropIdx = 64
    public static let conditionImageArea = 384 * 384

    public init(
        model: Qwen25VLModel, vision: QVLVision.VisionModel,
        tokenizer: any Tokenizers.Tokenizer, minPixels: Int, maxPixels: Int,
        dtype: DType = .bfloat16
    ) throws {
        self.model = model
        self.vision = vision
        self.tokenizer = tokenizer
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.dtype = dtype
        guard let pad = tokenizer.convertTokenToId("<|image_pad|>") else {
            throw QwenImageEditError.loading("tokenizer lacks <|image_pad|>")
        }
        self.imagePadId = pad
    }

    /// Load from a 2511 snapshot root (`text_encoder/` + `processor/`).
    ///
    /// `bits` (4/8) quantizes the VL-7B text model after load (the conditioning bulk; the
    /// small vision tower is left full precision). The model is eval'd immediately so the
    /// bf16 originals free before the caller loads the DiT — keeping the load peak down.
    public static func load(snapshot: URL, dtype: DType = .bfloat16, bits: Int? = nil) async throws
        -> QwenVLPromptEncoder
    {
        let encoderDir = snapshot.appendingPathComponent("text_encoder")
        let processorDir = snapshot.appendingPathComponent("processor")

        // --- config (flat HF layout; force tied so no lm_head module is created) ---
        let configData = try Data(contentsOf: encoderDir.appendingPathComponent("config.json"))
        var configJSON = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        configJSON["tie_word_embeddings"] = true
        guard let visionJSON = configJSON["vision_config"] as? [String: Any] else {
            throw QwenImageEditError.loading("text_encoder config has no vision_config")
        }
        let textConfig = try JSONDecoder().decode(
            Qwen25VLTextConfig.self,
            from: JSONSerialization.data(withJSONObject: configJSON))
        let model = Qwen25VLModel(textConfig)

        // --- weights: HF prefixes `model.*` (text) and `visual.*` (ViT) ---
        let files = try FileManager.default.contentsOfDirectory(
            at: encoderDir, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        var llm: [String: MLXArray] = [:]
        var vit: [String: MLXArray] = [:]
        for f in files {
            for (k, v) in try MLX.loadArrays(url: f) {
                if k.hasPrefix("model.") {
                    llm[String(k.dropFirst("model.".count))] = v.asType(dtype)
                } else if k.hasPrefix("visual.") {
                    vit[String(k.dropFirst("visual.".count))] = v.asType(dtype)
                } else if k == "lm_head.weight" {
                    continue  // untied head; unused for feature extraction
                } else {
                    throw QwenImageEditError.loading("unexpected key prefix: \(k)")
                }
            }
        }
        try QwenImageEditWeights.verifyAndLoad(model: model, weights: llm, label: "VL-7B")
        if let bits {
            quantize(model: model, groupSize: 64, bits: bits)
            eval(model)  // materialize int4 + free the bf16 originals now
        }

        // --- vision tower ---
        var visionDict = visionJSON
        if let inChans = visionDict.removeValue(forKey: "in_chans") {
            visionDict["in_channels"] = inChans
        }
        visionDict["model_type"] = visionDict["model_type"] ?? "qwen2_5_vl"
        let visionConfig = try JSONDecoder().decode(
            Qwen25VLConfiguration.VisionConfiguration.self,
            from: JSONSerialization.data(withJSONObject: visionDict))
        let vision = QVLVision.VisionModel(visionConfig)
        let sanitized = vision.sanitize(weights: vit)
        try vision.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys])
        eval(vision)

        // --- processor ---
        let processor = try Qwen25VLProcessorConfig.load(from: processorDir)
        let tokenizer = try await AutoTokenizer.from(modelFolder: processorDir)
        return try QwenVLPromptEncoder(
            model: model, vision: vision, tokenizer: tokenizer,
            minPixels: processor.minPixels, maxPixels: processor.maxPixels, dtype: dtype)
    }

    /// diffusers calculate_dimensions: sqrt-area ratio-preserved, /32 (Python banker's
    /// rounding) — returns (width, height).
    public static func calculateDimensions(targetArea: Int, ratio: Double) -> (Int, Int) {
        let width = (Double(targetArea) * ratio).squareRoot()
        let height = width / ratio
        let w32 = Int((width / 32).rounded(.toNearestOrEven)) * 32
        let h32 = Int((height / 32).rounded(.toNearestOrEven)) * 32
        return (w32, h32)
    }

    /// Encode (prompt, conditioning images) -> (1, S, 3584) prompt embeddings, the
    /// drop_idx-stripped last hidden state. Images are interleaved RGB8 buffers.
    public func encode(
        prompt: String, images: [(rgb: [UInt8], width: Int, height: Int)]
    ) throws -> MLXArray {
        // 1. Conditioning resize chain per image: LANCZOS to 384²-area /32 size.
        var processed: [(MLXArray, THW)] = []
        for image in images {
            let ratio = Double(image.width) / Double(image.height)
            let (cw, ch) = Self.calculateDimensions(
                targetArea: Self.conditionImageArea, ratio: ratio)
            let lanczos = PILLanczosResize.resize(
                rgb: image.rgb, width: image.width, height: image.height,
                outWidth: cw, outHeight: ch)
            // 2. HF processor: BICUBIC smart-resize (factor 28) + normalize + patchify.
            let (th, tw) = try Qwen25VLImageProcessing.targetSize(
                height: ch, width: cw,
                factor: Qwen25VLImageProcessing.patchSize * Qwen25VLImageProcessing.mergeSize,
                minPixels: minPixels, maxPixels: maxPixels)
            let bicubic = PILResize.resize(
                rgb: lanczos, width: cw, height: ch, outWidth: tw, outHeight: th)
            var chw = [Float](repeating: 0, count: 3 * th * tw)
            let plane = th * tw
            let mean = Qwen25VLImageProcessing.imageMean
            let std = Qwen25VLImageProcessing.imageStd
            for i in 0..<plane {
                let p = i * 3
                chw[i] = (Float(bicubic[p]) / 255 - mean[0]) / std[0]
                chw[plane + i] = (Float(bicubic[p + 1]) / 255 - mean[1]) / std[1]
                chw[2 * plane + i] = (Float(bicubic[p + 2]) / 255 - mean[2]) / std[2]
            }
            let frame = MLXArray(chw, [1, 3, th, tw])
            processed.append(try Qwen25VLImageProcessing.patchify(images: [frame]))
        }

        // 3. Edit-plus template with Picture prefixes.
        var imageBlock = ""
        for i in 0..<images.count {
            imageBlock += "Picture \(i + 1): <|vision_start|><|image_pad|><|vision_end|>"
        }
        let text = Self.promptTemplatePrefix + imageBlock + prompt + Self.promptTemplateSuffix
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)

        // 4. ViT features + pad expansion per image.
        let visionDtype = dtype
        let merge = Qwen25VLImageProcessing.mergeSize
        var features: [MLXArray] = []
        for (patches, frame) in processed {
            let f = vision(patches.asType(visionDtype), frames: [frame])
            features.append(f)
            let padCount = frame.product / (merge * merge)
            guard let idx = ids.firstIndex(of: imagePadId) else {
                throw QwenImageEditError.invalidInput("missing <|image_pad|> for image")
            }
            // Expand THIS pad occurrence only; later pads belong to later images.
            ids.replaceSubrange(
                idx...idx, with: Array(repeating: -imagePadId, count: padCount))
        }
        ids = ids.map { $0 == -imagePadId ? imagePadId : $0 }

        // 5. Merge features at pad positions (slice + concat).
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)
        var embeds = model.embedTokens(inputIds)
        let allFeatures = concatenated(features, axis: 0).asType(embeds.dtype)
        let padPositions = ids.enumerated().filter { $0.element == imagePadId }.map(\.offset)
        guard padPositions.count == allFeatures.dim(0) else {
            throw QwenImageEditError.invalidInput(
                "pad count \(padPositions.count) != ViT tokens \(allFeatures.dim(0))")
        }
        embeds = try Self.mergeImageFeatures(
            textEmbeds: embeds, imageFeatures: allFeatures, padPositions: padPositions)

        // 6. Positions: the diffusers reference runs the VL model WITHOUT
        // mm_token_type_ids, so it falls back to PLAIN SEQUENTIAL positions on all
        // three mRoPE axes (= standard 1D RoPE) — verified against the true SDPA
        // inputs (monkeypatch capture: q/k max_abs 0.0 vs sequential, 90.2 vs the
        // mRoPE grid). The 2511 DiT consumes embeddings produced this way; the
        // mRoPE-grid variant is NOT what the reference pipeline computes.
        let positionIds = Self.sequentialPositionIds(count: ids.count)

        // 7. Full causal forward; output = final-norm last hidden state.
        let hidden = model(
            inputEmbeddings: embeds, positionIds: positionIds, mask: .causal, caches: nil)
        return hidden[0..., Self.dropIdx..., 0...]
    }

    /// Slice + concatenate merge (single contiguous block per image, blocks in order).
    static func mergeImageFeatures(
        textEmbeds: MLXArray, imageFeatures: MLXArray, padPositions: [Int]
    ) throws -> MLXArray {
        // The pad positions form contiguous runs; features are concatenated in image
        // order, so a single pass over runs works.
        var parts: [MLXArray] = []
        var cursor = 0
        var featCursor = 0
        var i = 0
        while i < padPositions.count {
            var j = i
            while j + 1 < padPositions.count, padPositions[j + 1] == padPositions[j] + 1 {
                j += 1
            }
            let start = padPositions[i]
            let count = j - i + 1
            if start > cursor { parts.append(textEmbeds[0..., cursor..<start, 0...]) }
            parts.append(imageFeatures[featCursor..<(featCursor + count), 0...][.newAxis])
            cursor = start + count
            featCursor += count
            i = j + 1
        }
        if cursor < textEmbeds.dim(1) { parts.append(textEmbeds[0..., cursor..., 0...]) }
        return concatenated(parts, axis: 1)
    }

    /// Sequential positions broadcast to all 3 mRoPE axes (the reference behavior
    /// for the edit-plus encoder call — see encode() step 6).
    static func sequentialPositionIds(count: Int) -> MLXArray {
        let pos = MLXArray((0..<count).map { Int32($0) })
        return broadcast(pos.reshaped(1, 1, count), to: [3, 1, count])
    }

    /// 3D mRoPE position ids over text + multiple image grids (HF get_rope_index).
    /// NOT used by the reference edit-plus path — kept for reference/diagnostics.
    static func positionIds(
        ids: [Int], frames: [THW], mergeSize: Int, imagePadId: Int
    ) -> MLXArray {
        var t = [Int32](); var h = [Int32](); var w = [Int32]()
        var textPos: Int32 = 0
        var frameIdx = 0
        var i = 0
        while i < ids.count {
            if ids[i] == imagePadId {
                let frame = frames[frameIdx]
                frameIdx += 1
                let gridH = frame.h / mergeSize
                let gridW = frame.w / mergeSize
                let anchor = textPos
                for ti in 0..<frame.t {
                    for hi in 0..<gridH {
                        for wi in 0..<gridW {
                            t.append(anchor + Int32(ti))
                            h.append(anchor + Int32(hi))
                            w.append(anchor + Int32(wi))
                        }
                    }
                }
                i += frame.t * gridH * gridW
                textPos = anchor + Int32(max(frame.t, max(gridH, gridW)))
            } else {
                t.append(textPos); h.append(textPos); w.append(textPos)
                textPos += 1
                i += 1
            }
        }
        return stacked([MLXArray(t), MLXArray(h), MLXArray(w)], axis: 0)
            .reshaped(3, 1, ids.count)
    }
}

/// PIL-exact LANCZOS resize — same fixed-point machinery as Qwen25VL's PILResize
/// (Pillow Resample.c 8bpc), with the lanczos kernel: support 3.0,
/// sinc(x)·sinc(x/3). Used for the diffusers VaeImageProcessor conditioning resize.
public enum PILLanczosResize {
    static let precisionBits = 32 - 8 - 2

    static func sinc(_ x: Double) -> Double {
        if x == 0 { return 1 }
        let p = Double.pi * x
        return sin(p) / p
    }

    static func lanczos(_ xIn: Double) -> Double {
        let x = abs(xIn)
        if x < 3 { return sinc(x) * sinc(x / 3) }
        return 0
    }

    static func coefficients(inSize: Int, outSize: Int)
        -> (bounds: [(min: Int, count: Int)], coeffs: [[Int32]])
    {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 3.0 * filterscale
        let one = Double(1 << precisionBits)

        var bounds: [(Int, Int)] = []
        var coeffs: [[Int32]] = []
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            let count = xmax - xmin

            var w = [Double](repeating: 0, count: count)
            var total = 0.0
            for x in 0..<count {
                let v = lanczos((Double(x + xmin) - center + 0.5) / filterscale)
                w[x] = v
                total += v
            }
            var k = [Int32](repeating: 0, count: count)
            for x in 0..<count {
                let normalized = total != 0 ? w[x] / total : w[x]
                let scaled = normalized * one
                k[x] = Int32(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            bounds.append((xmin, count))
            coeffs.append(k)
        }
        return (bounds, coeffs)
    }

    @inline(__always)
    static func clip8(_ v: Int32) -> UInt8 {
        let shifted = v >> Int32(precisionBits)
        return UInt8(min(max(shifted, 0), 255))
    }

    public static func resize(
        rgb: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int
    ) -> [UInt8] {
        let half = Int32(1 << (precisionBits - 1))

        let (hBounds, hCoeffs) = coefficients(inSize: width, outSize: outWidth)
        var temp = [UInt8](repeating: 0, count: height * outWidth * 3)
        rgb.withUnsafeBufferPointer { src in
            temp.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let rowIn = y * width * 3
                    let rowOut = y * outWidth * 3
                    for xx in 0..<outWidth {
                        let (xmin, count) = hBounds[xx]
                        let k = hCoeffs[xx]
                        var s0 = half, s1 = half, s2 = half
                        for x in 0..<count {
                            let p = rowIn + (xmin + x) * 3
                            let w = k[x]
                            s0 += Int32(src[p]) * w
                            s1 += Int32(src[p + 1]) * w
                            s2 += Int32(src[p + 2]) * w
                        }
                        let o = rowOut + xx * 3
                        dst[o] = clip8(s0)
                        dst[o + 1] = clip8(s1)
                        dst[o + 2] = clip8(s2)
                    }
                }
            }
        }

        let (vBounds, vCoeffs) = coefficients(inSize: height, outSize: outHeight)
        var out = [UInt8](repeating: 0, count: outHeight * outWidth * 3)
        temp.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0..<outHeight {
                    let (ymin, count) = vBounds[yy]
                    let k = vCoeffs[yy]
                    let rowOut = yy * outWidth * 3
                    for xx in 0..<outWidth {
                        let col = xx * 3
                        var s0 = half, s1 = half, s2 = half
                        for y in 0..<count {
                            let p = (ymin + y) * outWidth * 3 + col
                            let w = k[y]
                            s0 += Int32(src[p]) * w
                            s1 += Int32(src[p + 1]) * w
                            s2 += Int32(src[p + 2]) * w
                        }
                        let o = rowOut + col
                        dst[o] = clip8(s0)
                        dst[o + 1] = clip8(s1)
                        dst[o + 2] = clip8(s2)
                    }
                }
            }
        }
        return out
    }
}
