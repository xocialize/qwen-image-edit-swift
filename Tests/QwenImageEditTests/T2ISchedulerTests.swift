// Offline gates for the T2I deltas — no weights, no Metal. The sigma schedule is the
// one thing a 4-step distilled model cannot be wrong about: the DMD2 student was trained
// on exactly this trajectory, so an off-by-one shift silently degrades every image.
//
// Run: swift test --filter T2ISchedulerTests

import MLX
import XCTest

@testable import QwenImageEdit

final class T2ISchedulerTests: XCTestCase {

    /// nvidia/Qwen-Image-Flash model card: "The packaged static shift-3 FlowMatch Euler
    /// scheduler produces effective sigmas [1.0, 0.9, 0.75, 0.5, 0.0] over the required
    /// four steps."
    func testFlashFourStepSigmas() {
        let sigmas = QwenImagePipeline.staticShiftedSigmas(steps: 4, shift: 3.0)
        let expected: [Float] = [1.0, 0.9, 0.75, 0.5, 0.0]
        XCTAssertEqual(sigmas.count, expected.count)
        for (got, want) in zip(sigmas, expected) {
            XCTAssertEqual(got, want, accuracy: 1e-6)
        }
    }

    /// Static shifting is monotonically decreasing and bracketed by [0, 1] at any step
    /// count — the reference formula shift*s / (1 + (shift-1)*s) on linspace(1, 1/n, n).
    func testStaticShiftIsMonotoneAndBracketed() {
        for steps in [1, 2, 8, 20, 50] {
            let sigmas = QwenImagePipeline.staticShiftedSigmas(steps: steps, shift: 3.0)
            XCTAssertEqual(sigmas.count, steps + 1)
            XCTAssertEqual(sigmas.first!, 1.0, accuracy: 1e-6)
            XCTAssertEqual(sigmas.last!, 0.0)
            for i in 1..<sigmas.count {
                XCTAssertLessThan(sigmas[i], sigmas[i - 1], "steps=\(steps) at i=\(i)")
            }
        }
    }

    /// shift == 1 is the identity, so the schedule collapses to plain linspace(1, 1/n, n).
    func testShiftOneIsIdentity() {
        let sigmas = QwenImagePipeline.staticShiftedSigmas(steps: 4, shift: 1.0)
        for (got, want) in zip(sigmas, [Float(1.0), 0.75, 0.5, 0.25, 0.0]) {
            XCTAssertEqual(got, want, accuracy: 1e-6)
        }
    }

    /// The static schedule must NOT be the dynamic one: Flash's scheduler_config.json sets
    /// `use_dynamic_shifting: false`, so mu/calculateShift play no part. This pins the two
    /// apart at the production sequence length (1024² -> 4096 tokens).
    func testStaticDiffersFromDynamic() {
        let mu = QwenImagePipeline.calculateShift(imageSeqLen: 4096)
        let dynamic = QwenImagePipeline.shiftedSigmas(steps: 4, mu: mu)
        let stat = QwenImagePipeline.staticShiftedSigmas(steps: 4, shift: 3.0)
        XCTAssertNotEqual(dynamic[1], stat[1], accuracy: 1e-3)
    }

    /// diffusers rounds the requested size down to a multiple of vae_scale_factor*2 = 16.
    func testRoundedSize() {
        XCTAssertEqual(
            QwenImageT2IGenerator.roundedSize(width: 1024, height: 1024).width, 1024)
        let odd = QwenImageT2IGenerator.roundedSize(width: 1000, height: 1500)
        XCTAssertEqual(odd.width, 992)
        XCTAssertEqual(odd.height, 1488)
    }

    /// The T2I path feeds the DiT a single image grid, so the zero_cond_t per-token
    /// modulation index is nil and every image token must take the real-t params — the
    /// reference `zero_cond_t=false` branch. This checks the block's shape contract on
    /// that path (a doubled temb against a batch-1 token stream used to broadcast to 2).
    func testBlockNilIndexUsesRealTHalf() {
        let dim = 64
        let block = QwenTransformerBlock(dim: dim, numAttentionHeads: 2, attentionHeadDim: 32)
        eval(block)
        let imgLen = 9
        let txtLen = 5
        let hidden = MLXArray.zeros([1, imgLen, dim])
        let encoder = MLXArray.zeros([1, txtLen, dim])
        let temb = MLXArray.zeros([2, dim])  // [temb(t), temb(0)]
        // RoPE tables: [seq, headDim/2] per stream (img then txt), as QwenEmbedRope emits.
        let imgRope = (MLXArray.ones([imgLen, 16]), MLXArray.zeros([imgLen, 16]))
        let txtRope = (MLXArray.ones([txtLen, 16]), MLXArray.zeros([txtLen, 16]))
        let (outTxt, outImg) = block(
            hiddenStates: hidden, encoderHiddenStates: encoder, temb: temb,
            imageRotaryEmb: (imgRope, txtRope), attentionMask: nil,
            modulateIndex: nil)
        eval(outTxt, outImg)
        XCTAssertEqual(outImg.shape, [1, imgLen, dim])
        XCTAssertEqual(outTxt.shape, [1, txtLen, dim])
    }
}
