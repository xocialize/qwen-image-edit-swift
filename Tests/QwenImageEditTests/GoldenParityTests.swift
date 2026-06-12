// S1 gate: DiT step-0 forward vs the P2 PT fp32 goldens.
//
// Goldens: VideoResearch/qwen-image-edit-models/goldens (captured by
// qwen-image-edit-mlx/scripts/capture_pt_goldens.py — diffusers fp32 CPU).
// Regimes (calibrated on the Python overlay, same goldens):
//   fp32 CPU stream: cosine >= 0.9999   (defect discriminator)
//   bf16 GPU:        cosine >= 0.9985   (production dtype; M5 matmul noise)
//
// XCTest only — the SPM metallib workaround (mlx-swift_Cmlx.bundle in .build/debug)
// does not survive swift-testing's helper process.
//
// Run: QIE_PARITY=1 [QIE_FP32_CPU=1] swift test --filter GoldenParityTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class GoldenParityTests: XCTestCase {
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/goldens")
    static let modelDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")

    func testDiTStep0() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PARITY"] == "1",
            "set QIE_PARITY=1 to run (loads the 20B transformer)")
        let fp32CPU = ProcessInfo.processInfo.environment["QIE_FP32_CPU"] == "1"
        if fp32CPU {
            Device.setDefault(device: Device(.cpu))
        }

        let enc = try MLX.loadArrays(url: Self.goldens.appendingPathComponent("enc_prompt.safetensors"))
        let dit = try MLX.loadArrays(url: Self.goldens.appendingPathComponent("dit_step0.safetensors"))
        let metaData = try Data(
            contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let target = meta["target_size"] as! [Int]  // [w, h]
        let vae = meta["vae_size"] as! [Int]
        let sigma0 = (meta["timestep_step0"] as! NSNumber).floatValue

        let dtype: DType = fp32CPU ? .float32 : .bfloat16
        let model = try QwenImageEditWeights.loadDiTFromPT(
            directory: Self.modelDir.appendingPathComponent("transformer"), dtype: dtype)

        let hidden = dit["hidden_in"]!.asType(dtype)
        let imgShapes = [
            (1, target[1] / 16, target[0] / 16),
            (1, vae[1] / 16, vae[0] / 16),
        ]

        func branch(_ embedsKey: String, _ goldenKey: String) -> Float {
            let out = model(
                hiddenStates: hidden,
                encoderHiddenStates: enc[embedsKey]!.asType(dtype),
                encoderHiddenStatesMask: nil,  // golden masks are all-ones
                timestep: MLXArray([sigma0]),
                imgShapes: imgShapes)
            let ours = out.asType(.float32).flattened()
            let ref = dit[goldenKey]!.asType(.float32).flattened()
            let cos = sum(ours * ref) / (sqrt(sum(ours * ours)) * sqrt(sum(ref * ref)) + 1e-12)
            let mae = mean(abs(ours - ref))
            eval(cos, mae)
            print("\(goldenKey): cosine \(cos.item(Float.self)) mae \(mae.item(Float.self))")
            return cos.item(Float.self)
        }

        let gate: Float = fp32CPU ? 0.9999 : 0.9985
        let cosPos = branch("prompt_embeds", "out_pos")
        XCTAssertGreaterThanOrEqual(cosPos, gate, "positive branch under \(gate)")
        let cosNeg = branch("neg_embeds", "out_neg")
        XCTAssertGreaterThanOrEqual(cosNeg, gate, "negative branch under \(gate)")
    }
}
