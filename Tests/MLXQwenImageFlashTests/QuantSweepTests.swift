// Quant-recipe sweep — attribution first, recipe second.
//
// The first int4 attempt (int4 attn+FFN, int8 modulation, group 64, int4 VL encoder) measured
// DiT step-0 cos 0.9623 vs the fp32 oracle and rendered visibly soft/washed-out. Before
// reaching for a different recipe, split the error: the pipeline has TWO quantized components
// (DiT and the VL-7B text encoder) and they are separately measurable against the same
// goldens. Guessing which one to fix is how a quant tier gets three wasted conversions.
//
// ⚠ GPU stream only — a quantized forward under a CPU pin silently grinds for hours.
//
// Run: QIF_SWEEP=1 swift test --filter FlashQuantSweepTests

import Foundation
import MLX
import MLXToolKit
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashQuantSweepTests: XCTestCase {
    static let goldens = FlashQuantGateTests.goldens
    static let bf16Snapshot = URL(fileURLWithPath: FlashPackageTests.snapshot)
    static let quantSnapshot = FlashQuantGateTests.quantSnapshot

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).flattened()
        let y = b.asType(.float32).flattened()
        let c = sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)) + 1e-12)
        eval(c)
        return c.item(Float.self)
    }

    /// Component A: how much does the int4 VL-7B text encoder alone cost? Same prompt, same
    /// golden, only the encoder precision differs. bf16 measures 0.9999926 here.
    func testInt4TextEncoderPromptEmbeds() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIF_SWEEP"] == "1", "QIF_SWEEP=1")
        let ref = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("prompt_embeds.safetensors"))
        let metaData = try Data(
            contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let prompt = meta["prompt"] as! String

        let variant = ProcessInfo.processInfo.environment["QIF_ENC_FILE"]
            ?? "text_encoder/model-int4.safetensors"
        let path = Self.quantSnapshot.appendingPathComponent(variant)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: path.path), "missing \(path.path)")
        let encoder = try await QwenVLPromptEncoder.loadTextOnly(
            snapshot: Self.quantSnapshot, quantizedTextModelPath: path.path)
        let ours = try encoder.encodeText(prompt: prompt)
        let cos = Self.cosine(ours, ref["prompt_embeds"]!)
        print("[sweep] VL-7B \(variant) prompt embeds vs fp32 oracle: cosine \(cos)  "
            + "(bf16 encoder: 0.9999926)")
    }

    /// Component B: the DiT alone, with the ORACLE's own prompt embeddings injected — so the
    /// encoder is factored out entirely and this is pure DiT quantization error.
    func testQuantizedDiTStep0Isolated() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIF_SWEEP"] == "1", "QIF_SWEEP=1")
        let variant = ProcessInfo.processInfo.environment["QIF_DIT_FILE"]
            ?? "transformer/model-int4.safetensors"
        let url = Self.quantSnapshot.appendingPathComponent(variant)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.path), "missing \(url.path)")

        let dit = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("dit_step0.safetensors"))
        let enc = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("prompt_embeds.safetensors"))
        let metaData = try Data(
            contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let w = meta["width"] as! Int
        let h = meta["height"] as! Int
        let sigma0 = (meta["sigmas"] as! [NSNumber])[0].floatValue

        let model = try QwenImageEditWeights.loadQuantizedDiT(from: url)
        let out = model(
            hiddenStates: dit["hidden_in"]!.asType(.bfloat16),
            encoderHiddenStates: enc["prompt_embeds"]!.asType(.bfloat16),
            encoderHiddenStatesMask: nil,
            timestep: MLXArray([sigma0]),
            imgShapes: [(1, h / 16, w / 16)])
        let cos = Self.cosine(out, dit["out"]!)
        print("[sweep] DiT \(variant) step-0 vs fp32 oracle: cosine \(cos)  "
            + "(bf16: 0.99836)")
    }

    /// Convert an alternative DiT recipe for the sweep. QIF_RECIPE selects it:
    ///   int8      — 8-bit attn+FFN, 8-bit modulation, group 64
    ///   int4g32   — 4-bit attn+FFN at group 32 (finer scales), 8-bit modulation
    func testConvertVariant() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIF_SWEEP"] == "1", "QIF_SWEEP=1")
        guard let recipe = ProcessInfo.processInfo.environment["QIF_RECIPE"] else {
            throw XCTSkip("set QIF_RECIPE=int8|int4g32")
        }
        let (config, name): (QwenImageEditWeights.DiTQuantConfig, String)
        switch recipe {
        case "int8":
            config = .init(ditBits: 8, modulationBits: 8, groupSize: 64)
            name = "transformer/model-int8.safetensors"
        case "int4g32":
            config = .init(ditBits: 4, modulationBits: 8, groupSize: 32)
            name = "transformer/model-int4g32.safetensors"
        case "encoder-int8":
            // The encoder is the OTHER quantized component; at int4 it costs 0.9845 on the
            // prompt embeds, which is the dominant error term once the DiT is int8.
            let out = Self.quantSnapshot.appendingPathComponent(
                "text_encoder/model-int8.safetensors")
            if FileManager.default.fileExists(atPath: out.path) {
                print("[sweep] encoder int8 already exists"); return
            }
            let t = Date()
            try QwenVLPromptEncoder.saveQuantizedTextModel(
                snapshot: Self.bf16Snapshot, to: out, bits: 8, groupSize: 64)
            let mb = ((try? FileManager.default.attributesOfItem(atPath: out.path)[.size])
                as? Int ?? 0) / 1_000_000
            print("[sweep] wrote text_encoder/model-int8.safetensors (\(mb) MB) in "
                + String(format: "%.0f s", Date().timeIntervalSince(t)))
            return
        default:
            throw XCTSkip("unknown QIF_RECIPE \(recipe)")
        }
        let out = Self.quantSnapshot.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: out.path) {
            print("[sweep] \(name) already exists"); return
        }
        let t = Date()
        try QwenImageEditWeights.saveQuantizedDiT(
            from: Self.bf16Snapshot.appendingPathComponent("transformer"),
            to: out, config: config)
        let mb = ((try? FileManager.default.attributesOfItem(atPath: out.path)[.size]) as? Int
            ?? 0) / 1_000_000
        print("[sweep] wrote \(name) (\(mb) MB) in "
            + String(format: "%.0f s", Date().timeIntervalSince(t)))
    }
}
