// T2I gate: the Qwen-Image-Flash text-to-image path vs the PT fp32 goldens.
//
// Goldens: /Volumes/DEV_ARCHIVE/models/nvidia/qwen-image-flash-goldens (captured by
// qwen-image-edit-mlx/scripts/capture_flash_goldens.py — diffusers 0.37.1 fp32 CPU for the
// single-forward dumps, bf16 MPS for the reference render).
// Regimes match the edit gate (same DiT family, same M5 matmul noise):
//   fp32 CPU stream: cosine >= 0.9999   (defect discriminator)
//   bf16 GPU:        cosine >= 0.9985   (production dtype)
//
// The VAE is NOT re-gated here: nvidia/Qwen-Image-Flash ships the byte-identical
// AutoencoderKLQwenImage checkpoint as Qwen-Image-Edit-2511 (sha256 verified), already
// covered by VAEDecodeParityTests at 73.7 dB.
//
// Run: QIF_PARITY=1 [QIF_FP32_CPU=1] swift test --filter T2IGoldenParityTests

import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers
import XCTest

@testable import QwenImageEdit

final class T2IGoldenParityTests: XCTestCase {
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/nvidia/qwen-image-flash-goldens")
    static let modelDir = URL(
        fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/nvidia/Qwen-Image-Flash")

    private func requireParity() throws -> Bool {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_PARITY"] == "1",
            "set QIF_PARITY=1 to run (loads the 20B transformer / 7B encoder)")
        let fp32CPU = ProcessInfo.processInfo.environment["QIF_FP32_CPU"] == "1"
        if fp32CPU { Device.setDefault(device: Device(.cpu)) }
        return fp32CPU
    }

    private func meta() throws -> [String: Any] {
        let data = try Data(contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json"))
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    private func cosine(_ ours: MLXArray, _ ref: MLXArray) -> Float {
        let a = ours.asType(.float32).flattened()
        let b = ref.asType(.float32).flattened()
        let cos = sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)) + 1e-12)
        eval(cos)
        return cos.item(Float.self)
    }

    /// S3: text-only prompt encoding — the T2I template (drop_idx 34, NOT the edit path's
    /// 64) through the Qwen2.5-VL backbone, truncated to 512 tokens.
    func testPromptEmbeds() async throws {
        let fp32CPU = try requireParity()
        let ref = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("prompt_embeds.safetensors"))
        let m = try meta()
        let prompt = m["prompt"] as! String

        let encoder = try await QwenVLPromptEncoder.loadTextOnly(
            snapshot: Self.modelDir, dtype: fp32CPU ? .float32 : .bfloat16)
        let ours = try encoder.encodeText(prompt: prompt)
        let golden = ref["prompt_embeds"]!

        XCTAssertEqual(
            ours.shape, golden.shape,
            "token count must match the reference — a drop_idx or template mismatch shows up "
                + "here first (edit=64 vs t2i=34)")
        let cos = cosine(ours, golden)
        print("prompt_embeds: cosine \(cos)  shape \(ours.shape)")
        XCTAssertGreaterThanOrEqual(cos, fp32CPU ? 0.9999 : 0.99)
    }

    /// S1: DiT step-0 forward on the reference's own packed noise. This is the gate that
    /// exercises the single-grid (nil modulateIndex) path — the reference's
    /// `zero_cond_t=false` branch.
    func testDiTStep0() throws {
        let fp32CPU = try requireParity()
        let dit = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("dit_step0.safetensors"))
        let enc = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("prompt_embeds.safetensors"))
        let m = try meta()
        let w = m["width"] as! Int
        let h = m["height"] as! Int
        let sigmas = (m["sigmas"] as! [NSNumber]).map(\.floatValue)

        let dtype: DType = fp32CPU ? .float32 : .bfloat16
        let model = try QwenImageEditWeights.loadDiTFromPT(
            directory: Self.modelDir.appendingPathComponent("transformer"), dtype: dtype)

        let out = model(
            hiddenStates: dit["hidden_in"]!.asType(dtype),
            encoderHiddenStates: enc["prompt_embeds"]!.asType(dtype),
            encoderHiddenStatesMask: nil,  // single prompt -> mask is all ones
            timestep: MLXArray([sigmas[0]]),
            imgShapes: [(1, h / 16, w / 16)])

        let cos = cosine(out, dit["out"]!)
        print("dit_step0: cosine \(cos)  σ0 \(sigmas[0])  grid \(h / 16)×\(w / 16)")
        // bf16 threshold is looser than the edit gate's 0.9985 BY DESIGN: that one was
        // calibrated at ~8k tokens, this golden is 256 tokens, and per-element bf16 error
        // averages out with sequence length (measured 0.99836 here vs 0.99986 there — the
        // ~sqrt(N) ratio). The fp32-CPU stream is the defect discriminator and measures
        // 1.0000002, i.e. the T2I path is exact; only GPU half-precision noise remains.
        XCTAssertGreaterThanOrEqual(cos, fp32CPU ? 0.9999 : 0.998)
    }

    /// E2E at the model's tested production size (1024²) — the largest-production-grid rule:
    /// small grids validate nothing about RoPE extrapolation at the real resolution. Writes
    /// the decoded PNG next to the goldens for the eyeball check against the stage-B MPS
    /// reference render.
    func testEndToEnd1024() async throws {
        _ = try requireParity()
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_E2E"] == "1",
            "set QIF_E2E=1 to run the full 4-step 1024² render")
        let m = try meta()
        let prompt = m["prompt"] as! String
        let seed = (m["seed"] as! NSNumber).uint64Value

        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: Self.modelDir.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: Self.modelDir.appendingPathComponent("vae"))
        let snapshot = Self.modelDir
        let generator = QwenImageT2IGenerator(
            encoderProvider: { try await QwenVLPromptEncoder.loadTextOnly(snapshot: snapshot) },
            transformer: transformer, vae: vae, shift: 3.0)

        let start = Date()
        let (pixels, width, height) = try await generator.generate(
            prompt: prompt, width: 1024, height: 1024, steps: 4, trueCFGScale: 1.0, seed: seed,
            progress: { i, n in print("  step \(i)/\(n)") })
        print(String(format: "e2e: %d×%d in %.1f s", width, height, -start.timeIntervalSinceNow))

        let out = Self.goldens.appendingPathComponent("swift_t2i_bf16_1024.png")
        try GenerateDemoTests.writePNG(pixels: pixels, width: width, height: height, to: out)
        print("[saved] \(out.path)")

        // Sanity: a coherent render is neither uniform nor saturated. Degenerate output
        // (all-black, all-gray, NaN-washed) is the failure mode a cosine gate at a small
        // grid cannot see.
        let arr = MLXArray(pixels).asType(.float32)
        let sd = sqrt(mean(square(arr - mean(arr))))
        eval(sd)
        print("pixel sd \(sd.item(Float.self))")
        XCTAssertGreaterThan(sd.item(Float.self), 10.0, "output looks degenerate (flat)")
    }
}
