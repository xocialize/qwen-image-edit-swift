// Correctness gate for QwenImageEditLoRASwapper (dropdown hot-swap): switching adapters on a
// resident DiT must restore the pristine base exactly, so a swapped render is bit-identical to
// a freshly-loaded one. Path: load int4 DiT into a swapper, set [Lightning+anime] and render,
// then set [Lightning+pixar] and render — compare that to pixar applied stateless on a fresh
// DiT. Any residue from incomplete detach would show as a non-zero pixel MAD.
//
// Run: QIE_SWAP=1 swift test --filter LoRASwapTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class LoRASwapTests: XCTestCase {
    func testSwapRestoresBaseExactly() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIE_SWAP"] == "1", "QIE_SWAP=1")
        typealias C = LoRACommunityRenderTests
        for p in [C.quantizedDiT, C.quantizedEnc, C.lightning.path, C.anime.path, C.pixar.path,
                  C.input.path] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }

        let image = try EncoderParityTests.loadRGB(url: C.input)
        let encoder = try await QwenVLPromptEncoder.load(
            snapshot: C.modelDir, quantizedTextModelPath: C.quantizedEnc)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: C.modelDir.appendingPathComponent("vae"), dtype: .bfloat16)
        let pixarPrompt = "Transform it into Pixar-inspired 3D"

        func render(_ dit: QwenImageTransformer2DModel, _ prompt: String) async throws -> [UInt8] {
            try await QwenImageEditGenerator(encoder: encoder, transformer: dit, vae: vae)
                .generate(image: image, prompt: prompt, negativePrompt: " ",
                          steps: 4, trueCFGScale: 1.0, seed: 42, progress: { _, _ in }).pixels
        }
        func freshDiT() throws -> QwenImageTransformer2DModel {
            try QwenImageEditWeights.loadQuantizedDiT(from: URL(fileURLWithPath: C.quantizedDiT))
        }

        // Swapped path: anime first, then swap to pixar on the SAME resident DiT. The swapper
        // mutates `dit` in place, so `dit` is always the currently-adapted model.
        let dit = try freshDiT()
        let swapper = QwenImageEditLoRASwapper(model: dit)
        try swapper.set([(C.lightning, 4.0), (C.anime, 1.0)])
        _ = try await render(dit, "transform into anime")
        try swapper.set([(C.lightning, 4.0), (C.pixar, 1.0)])
        let swapped = try await render(dit, pixarPrompt)
        GPU.clearCache()

        // Reference: pixar applied stateless on a fresh DiT.
        let freshPixar = try freshDiT()
        try QwenImageEditLoRA.apply(diffusersLoRAs: [(C.lightning, 4.0), (C.pixar, 1.0)], to: freshPixar)
        let reference = try await render(freshPixar, pixarPrompt)

        XCTAssertEqual(swapped.count, reference.count)
        var maxAbs = 0, sum = 0
        for i in 0..<min(swapped.count, reference.count) {
            let d = abs(Int(swapped[i]) - Int(reference[i]))
            maxAbs = max(maxAbs, d); sum += d
        }
        let mad = Double(sum) / Double(swapped.count)
        print("[swap] swapped-vs-fresh pixar: MAD=\(mad) maxAbs=\(maxAbs) over \(swapped.count) px-bytes")
        // Same weights + same seed => identical; allow only trivial nondeterminism slack.
        XCTAssertEqual(maxAbs, 0, "swap left residue (maxAbs \(maxAbs)); detach incomplete")
    }
}
