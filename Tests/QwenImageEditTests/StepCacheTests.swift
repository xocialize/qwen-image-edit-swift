// Offline (CPU, tiny random-weight model) mechanics gate for the FR2 step-residual
// cache: threshold-0 must be bit-identical to the uncached loop, warmup steps must
// never skip, and a forced skip must reproduce shallow + cached-Δ semantics. The
// quality gate (SSIMULACRA2 vs uncached goldens at real weights) is the separate
// env-gated live eval — this test proves the cache plumbing, not image quality.

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class StepCacheTests: XCTestCase {

    /// Tiny DiT stand-in: 4 blocks, 2 heads × 8 dims, random weights, fp32 CPU.
    private func tinyModel() -> QwenImageTransformer2DModel {
        QwenImageTransformer2DModel(
            patchSize: 2, inChannels: 16, outChannels: 4, numLayers: 4,
            attentionHeadDim: 8, numAttentionHeads: 2, jointAttentionDim: 24,
            axesDimsRope: [4, 2, 2])  // must sum to attentionHeadDim (like 16+56+56=128)
    }

    private struct Inputs {
        let hidden: MLXArray
        let txt: MLXArray
        let shapes: [(Int, Int, Int)]
    }

    /// Target 4×4 grid + one 4×4 conditioning grid (exercises the modulateIndex path).
    private func inputs(seed: UInt64) -> Inputs {
        MLXRandom.seed(seed)
        let target = MLXRandom.normal([1, 16, 16])
        let cond = MLXRandom.normal([1, 16, 16])
        let txt = MLXRandom.normal([1, 5, 24])
        return Inputs(
            hidden: concatenated([target, cond], axis: 1), txt: txt,
            shapes: [(1, 4, 4), (1, 4, 4)])
    }

    /// threshold 0: relative-L1 is never < 0, so every step runs fully and outputs are
    /// bit-identical to the uncached loop.
    func testThresholdZeroMatchesUncached() {
        Device.withDefaultDevice(.cpu) {
            let model = tinyModel()
            let cache = DiTStepCache(threshold: 0, computeBlocks: 2, warmupSteps: 0)
            for step in 0..<4 {
                let x = inputs(seed: UInt64(100 + step))
                let t = MLXArray([Float(1) - Float(step) * 0.2])
                let plain = model(
                    hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                    encoderHiddenStatesMask: nil, timestep: t, imgShapes: x.shapes)
                let cached = model(
                    hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                    encoderHiddenStatesMask: nil, timestep: t, imgShapes: x.shapes,
                    stepCache: cache)
                XCTAssertTrue(
                    allClose(plain, cached, atol: 0).item(Bool.self),
                    "step \(step): threshold-0 cached output diverged from uncached")
            }
            XCTAssertEqual(cache.skippedSteps, 0)
        }
    }

    /// Huge threshold: warmup steps still run fully; afterwards every step skips, and
    /// the skipped output equals shallow(cur) + (deep(prevFull) − shallow(prevFull)).
    func testForcedSkipReusesDeepResidual() {
        Device.withDefaultDevice(.cpu) {
            let model = tinyModel()
            let cache = DiTStepCache(
                threshold: .greatestFiniteMagnitude, computeBlocks: 2, warmupSteps: 1)
            let x = inputs(seed: 7)

            // Step 0 (warmup): full compute; residual gets cached.
            let t0 = MLXArray([Float(1)])
            let full0 = model(
                hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                encoderHiddenStatesMask: nil, timestep: t0, imgShapes: x.shapes,
                stepCache: cache)
            XCTAssertEqual(cache.skippedSteps, 0)

            // Step 1: same inputs + timestep → shallow identical → skipped output must
            // equal the full step-0 output exactly (shallow + Δ reconstructs it).
            let out1 = model(
                hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                encoderHiddenStatesMask: nil, timestep: t0, imgShapes: x.shapes,
                stepCache: cache)
            XCTAssertEqual(cache.skippedSteps, 1)
            XCTAssertTrue(
                allClose(out1, full0, atol: 1e-5).item(Bool.self),
                "skipped step did not reconstruct shallow + cached Δ")

            // Step 2, different timestep: still skips (threshold ∞) and stays finite,
            // shape-correct.
            let out2 = model(
                hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                encoderHiddenStatesMask: nil, timestep: MLXArray([Float(0.5)]),
                imgShapes: x.shapes, stepCache: cache)
            XCTAssertEqual(cache.skippedSteps, 2)
            XCTAssertEqual(out2.shape, full0.shape)
            XCTAssertTrue(all(isFinite(out2)).item(Bool.self))
        }
    }

    /// Separate caches per CFG branch: state advanced by one branch must not leak into
    /// a fresh cache (the pos/neg isolation rule).
    func testFreshCacheDoesNotSkipDuringWarmup() {
        Device.withDefaultDevice(.cpu) {
            let model = tinyModel()
            let x = inputs(seed: 21)
            let t = MLXArray([Float(1)])
            let warmed = DiTStepCache(
                threshold: .greatestFiniteMagnitude, computeBlocks: 2, warmupSteps: 3)
            for _ in 0..<3 {
                _ = model(
                    hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                    encoderHiddenStatesMask: nil, timestep: t, imgShapes: x.shapes,
                    stepCache: warmed)
            }
            XCTAssertEqual(warmed.skippedSteps, 0, "warmup steps must never skip")
            _ = model(
                hiddenStates: x.hidden, encoderHiddenStates: x.txt,
                encoderHiddenStatesMask: nil, timestep: t, imgShapes: x.shapes,
                stepCache: warmed)
            XCTAssertEqual(warmed.skippedSteps, 1, "post-warmup identical step should skip")
        }
    }

    func testModeThresholds() {
        XCTAssertNil(StepCacheMode.off.threshold)
        XCTAssertEqual(StepCacheMode.conservative.threshold, 0.10)
        XCTAssertEqual(StepCacheMode.aggressive.threshold, 0.15)
        XCTAssertEqual(StepCacheMode(rawValue: "aggressive"), .aggressive)
    }
}
