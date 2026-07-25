// int8 tier gate (the shipping quantized tier; int4 was measured and rejected — see
// FlashQuantSweepTests).
//
// ⚠ These tests must NOT pin the CPU stream. Quantized matmul is Metal-only; under
// `Device.setDefault(.cpu)` a quantized forward has no efficient path and silently grinds for
// HOURS (state R, ~100% CPU, no output, no error, no watchdog). The fp32-CPU discriminator
// used by the bf16 parity gate is exactly the wrong move here. The GPU fp32 noise floor
// (~1e-3) is negligible against int4 error at a ≥0.99 gate.
//
// Gate is per-pass cosine on IDENTICAL injected inputs, not PSNR against a golden image:
// quantization perturbs the denoise trajectory into a different-but-equally-valid image, so
// an image-space metric reads as failure even when the tier is fine.
//
// Run: QIF_QUANT_GATE=1 swift test --filter FlashQuantGateTests

import Foundation
import MLX
import MLXToolKit
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashQuantGateTests: XCTestCase {
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/nvidia/qwen-image-flash-goldens")
    static let quantSnapshot = URL(
        fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/nvidia/Qwen-Image-Flash-8bit")

    /// int4 DiT step-0 vs the PT fp32 oracle, on the oracle's own packed noise and prompt
    /// embeddings. Deliberately measured against fp32 (not against our bf16 run), so the number
    /// carries BOTH the quantization error and the precision gap — a conservative gate. For
    /// reference the same forward in bf16 measures 0.99836 on this golden.
    func testInt8DiTStep0() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_QUANT_GATE"] == "1", "QIF_QUANT_GATE=1")
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

        let model = try QwenImageEditWeights.loadQuantizedDiT(
            from: Self.quantSnapshot.appendingPathComponent(
                QwenImageFlashConfiguration.int8DiTFile))
        let out = model(
            hiddenStates: dit["hidden_in"]!.asType(.bfloat16),
            encoderHiddenStates: enc["prompt_embeds"]!.asType(.bfloat16),
            encoderHiddenStatesMask: nil,
            timestep: MLXArray([sigma0]),
            imgShapes: [(1, h / 16, w / 16)])

        let ours = out.asType(.float32).flattened()
        let ref = dit["out"]!.asType(.float32).flattened()
        let cos = (sum(ours * ref) / (sqrt(sum(ours * ours)) * sqrt(sum(ref * ref)) + 1e-12))
            .item(Float.self)
        print("int8 dit_step0 vs fp32 oracle: cosine \(cos)  (bf16 reference: 0.99836)")
        XCTAssertGreaterThanOrEqual(cos, 0.99)
    }

    /// End-to-end int4 render at the production grid, through the package. Writes the PNG for
    /// the visual check — the third leg of the quant gate (cosine + image-validity + eyeball).
    func testInt8EndToEnd1024() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_QUANT_GATE"] == "1", "QIF_QUANT_GATE=1")
        let package = QwenImageFlashPackage(
            configuration: .init(snapshotPath: Self.quantSnapshot.path, quant: .int8))
        let loadStart = Date()
        try await package.load()
        print(String(format: "int8 load: %.1f s", -loadStart.timeIntervalSinceNow))

        let start = Date()
        let response = try await package.run(T2IRequest(
            prompt: "A red fox in a snowy pine forest at golden hour, photorealistic, "
                + "sharp focus, soft bokeh",
            width: 1024, height: 1024, seed: 42))
        guard let t2i = response as? T2IResponse else { return XCTFail("wrong response type") }
        print(String(format: "int8 e2e: 1024x1024 in %.1f s", -start.timeIntervalSinceNow))

        let out = Self.goldens.appendingPathComponent("swift_t2i_int8_1024.png")
        try t2i.image.data.write(to: out)
        print("[saved] \(out.path)")
        await package.unload()
    }

    /// Split footprint for the int4 tier — the numbers the manifest declares for the devices
    /// this tier exists to reach.
    func testInt8MemBench() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_QUANT_GATE"] == "1", "QIF_QUANT_GATE=1")
        let gb = 1_000_000_000.0
        let snapshot = Self.quantSnapshot

        let transformer = try QwenImageEditWeights.loadQuantizedDiT(
            from: snapshot.appendingPathComponent(QwenImageFlashConfiguration.int8DiTFile))
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        eval(transformer.parameters())
        eval(vae.parameters())
        MLX.GPU.clearCache()
        let floor = Double(MLX.GPU.activeMemory) / gb
        print(String(format: "[membench int8] resident floor (DiT+VAE): %.1f GB", floor))

        let encPath = snapshot.appendingPathComponent(
            QwenImageFlashConfiguration.int8TextEncoderFile).path
        let generator = QwenImageT2IGenerator(
            encoderProvider: {
                try await QwenVLPromptEncoder.loadTextOnly(
                    snapshot: snapshot, quantizedTextModelPath: encPath)
            },
            transformer: transformer, vae: vae, shift: 3.0)

        MLX.GPU.clearCache()
        MLX.GPU.resetPeakMemory()
        _ = try await generator.generate(
            prompt: "A red fox in a snowy pine forest at golden hour, photorealistic",
            width: 1024, height: 1024, steps: 4, trueCFGScale: 1.0, seed: 42,
            progress: { _, _ in })
        let peak = Double(MLX.GPU.peakMemory) / gb
        let activation = max(0, peak - floor)
        print(String(
            format: "[membench int8] 1024x1024 4-step | peak %.1f GB | floor %.1f GB | "
                + "activation %.1f GB", peak, floor, activation))
        print(String(
            format: "[membench int8] DECLARE -> residentBytes=%.0f  peakActivationBytes=%.0f "
                + "(+20%% -> %.0f)", floor * gb, activation * gb, activation * 1.2 * gb))
    }
}
