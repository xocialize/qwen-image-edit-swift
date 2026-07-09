// DiT step-residual caching (DBCache family — FireRed fast_pipeline / cache-dit
// DBCacheConfig(Fn_compute_blocks=8, Bn_compute_blocks=0, residual_diff_threshold,
// max_warmup_steps=3)).
//
// Adjacent denoise steps produce highly similar transformer activations. Each step we
// always compute the first `computeBlocks` transformer blocks; while the ACCUMULATED
// relative-L1 drift of their residual stays under `threshold`, the remaining deep blocks
// are SKIPPED and the cached deep residual (Δ = deepOut − shallowOut from the last full
// step) is reused: out ≈ shallow + Δ. Reaching the threshold forces a full compute and
// resets the drift budget (TeaCache-style accumulation — see reuse() for why prev-step-
// only comparison over-skips). Training-free, checkpoint-agnostic, orthogonal to
// quantization. Under true CFG the pos and neg branches run as separate forwards and
// need SEPARATE caches — never share a DiTStepCache across branches.

import Foundation
import MLX

/// Opt-in step-residual cache level for the denoise loop. Raw-string Codable so it can
/// ride PackageConfiguration and request metaData unchanged.
/// Both CFG branches are always cached together: their skip patterns then correlate and
/// the reconstruction errors partially cancel in the guidance difference (pos − neg).
/// The neg-only variant was measured WORSE (−9.4 vs 36.6 SSIMULACRA2) — `guidedNoise`
/// weights neg by (1 − scale) = −3 at scale 4, so an uncorrelated neg error is amplified
/// straight into the image.
///
/// **Measured (2026-07-09, 20-step true-CFG seed-42 A/B, accumulation + Taylor):**
/// conservative 36.6 SSIMULACRA2 / 7 skips per branch; aggressive −18.8 / 10 skips.
/// Both are far below the ≥85 default-on bar — at 20 steps the per-step deep-residual
/// drift is 6–11%, so the temporal-redundancy assumption fails. STAYS OPT-IN; revisit
/// only for ≥30–40-step schedules where per-step drift halves.
public enum StepCacheMode: String, Codable, Sendable, CaseIterable {
    case off
    case conservative  // drift budget 0.10
    case aggressive    // drift budget 0.15 (FireRed's shipped setting)

    /// Accumulated relative-L1 drift budget; nil = caching disabled.
    public var threshold: Float? {
        switch self {
        case .off: return nil
        case .conservative: return 0.10
        case .aggressive: return 0.15
        }
    }
}

/// Request-level metaData key for the step-residual cache (shared by the engine
/// wrappers; value = a StepCacheMode raw string). Declared in each surface descriptor
/// summary per the metaData governance rule.
public enum StepCacheMetaKeys {
    public static let mode = "stepCacheMode"
}

/// Per-branch (pos/neg) step-residual cache state. Create one per CFG branch per
/// generate() call — it is stateful across denoise steps and must not outlive the loop.
public final class DiTStepCache {
    public let threshold: Float
    /// Always-computed shallow prefix (DBCache Fn). 8 of the 2511 DiT's 60 blocks.
    public let computeBlocks: Int
    /// Steps that always run fully before skipping is considered (DBCache max_warmup_steps).
    public let warmupSteps: Int
    /// Diagnostics: how many steps reused the cached residual.
    public private(set) var skippedSteps = 0

    private var step = 0
    private var accumulated: Float = 0
    private var prevShallow: MLXArray?
    /// Last two fully-computed deep residuals with their sigmas, newest last — the
    /// support points for the first-order (TaylorSeer-style) Δ extrapolation.
    private var deltas: [(sigma: Float, delta: MLXArray)] = []

    public init(threshold: Float, computeBlocks: Int = 8, warmupSteps: Int = 3) {
        self.threshold = threshold
        self.computeBlocks = computeBlocks
        self.warmupSteps = warmupSteps
    }

    /// Decide this step from the shallow (block-`computeBlocks`) img-stream output and
    /// its RESIDUAL across those blocks (shallowOut − blockInput, the Fn contribution).
    /// Returns the reconstructed deep output (shallow + cached Δ) when the step can be
    /// skipped, nil when the deep blocks must run. Advances the per-step state either way.
    ///
    /// The relative-L1 deltas are ACCUMULATED across consecutive skips (TeaCache
    /// semantics) and a full compute is forced — resetting the accumulator — once the
    /// summed drift reaches `threshold`. Prev-step-only comparison is NOT enough: the
    /// per-step delta stays tiny on a smooth trajectory, so skipping never breaks and the
    /// deep-block Δ freezes at an early noisy step (live A/B: 15/20 skips, washed-out
    /// low-contrast output, SSIMULACRA2 −64; deep blocks are timestep-modulated, so their
    /// stale contribution is what drains the contrast). Accumulation bounds Δ staleness
    /// by construction — FireRed's config leans on a TaylorSeer calibrator for the same
    /// reason; the accumulator is the calibrator-free equivalent.
    public func reuse(shallow: MLXArray, residual: MLXArray, sigma: Float) -> MLXArray? {
        defer {
            step += 1
            prevShallow = residual
        }
        guard step >= warmupSteps, let prev = prevShallow, let delta = deltas.last
        else { return nil }
        // Relative L1 in fp32 (bf16 mean underflows the small deltas this thresholds on).
        let cur = residual.asType(.float32)
        let ref = prev.asType(.float32)
        let rel = (mean(abs(cur - ref)) / mean(abs(ref))).item(Float.self)
        accumulated += rel
        if ProcessInfo.processInfo.environment["QIE_STEPCACHE_DEBUG"] == "1" {
            print(String(
                format: "[stepcache] step %d σ %.3f rel %.4f acc %.4f -> %@",
                step, sigma, rel, accumulated, accumulated < threshold ? "skip" : "compute"))
        }
        guard accumulated < threshold else {
            accumulated = 0  // full compute refreshes Δ — drift budget restarts
            return nil
        }
        skippedSteps += 1
        // First-order Δ extrapolation along sigma (TaylorSeer calibrator, order 1): the
        // deep blocks are timestep-modulated, so their contribution moves with σ even
        // when the shallow trajectory is smooth — a FROZEN Δ measurably drains contrast
        // (live A/B at 20-step CFG: frozen Δ scored 7.0/−12.9 SSIMULACRA2 even with
        // accumulation-bounded skipping). Falls back to frozen Δ until two support
        // points exist.
        if deltas.count == 2, abs(deltas[1].sigma - deltas[0].sigma) > 1e-6 {
            let (s0, d0) = deltas[0]
            let (s1, d1) = deltas[1]
            let slope = (d1 - d0) * ((sigma - s1) / (s1 - s0))
            return shallow + d1 + slope.asType(shallow.dtype)
        }
        return shallow + delta.delta
    }

    /// Record a fully-computed step's deep residual (+ its sigma) as the newest Taylor
    /// support point. `eval` keeps the retained arrays materialized instead of pinning
    /// the whole step graph across steps.
    public func store(shallow: MLXArray, deep: MLXArray, sigma: Float) {
        let delta = deep - shallow
        eval(delta)
        deltas.append((sigma, delta))
        if deltas.count > 2 { deltas.removeFirst() }
    }
}
