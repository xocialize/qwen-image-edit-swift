// Progress.swift — core-owned ambient run-phase seam (AB-A-0139).
//
// The QwenImageEdit core stays engine-free (MLXToolKit is a WRAPPER dependency — see
// Package.swift), so the core reports its phases into its OWN task-local sink and each wrapper
// target (MLXQwenImageEdit, MLXQwenImageEditTurbo, MLXTeleStyle, MLXQwenImageFlash) forwards
// them into the engine contract's `RunProgress` (contract 1.18.0, ENGINE-NEEDS V2).
// Same shape as the contract plane: a coarse phase plus optional 1-based step/totalSteps. No-op
// when unbound, so the CLI and the parity gates run unchanged.
//
// Task-local (not a pipeline parameter) so a binding scopes to exactly one generation and the
// pipeline/generator signatures stay as they are — mirrors MageProgress (mage-flow-swift).

public enum QwenImageEditProgress {
    /// The phases a Qwen-Image run (edit and T2I) passes through. The raw values ARE the engine
    /// contract's canonical `RunPhase` constants, so the wrappers forward them with no mapping
    /// table.
    public enum Phase: String, Sendable {
        /// Qwen2.5-VL-7B conditioning, including the per-request encoder load (skipped on a
        /// prompt-memo hit), and the VAE encode of the conditioning images on the edit path.
        case encode
        /// The flow-matching Euler loop. Reported once at loop entry with no step, then per
        /// completed step — the run's dominant cost.
        case denoise
        /// Qwen-Image VAE decode to pixels, reported once (the bounded decode above the 1024
        /// bucket tiles internally; no per-chunk counts are reported).
        case decode
    }

    /// One coarse observation: the phase, plus 1-based step counts when the phase has steps.
    public struct Event: Sendable, Equatable {
        public var phase: Phase
        public var step: Int?
        public var totalSteps: Int?

        public init(phase: Phase, step: Int? = nil, totalSteps: Int? = nil) {
            self.phase = phase
            self.step = step
            self.totalSteps = totalSteps
        }
    }

    public typealias Sink = @Sendable (Event) -> Void

    @TaskLocal public static var sink: Sink?

    /// Report a phase observation to the ambient sink (no-op if none is bound).
    public static func report(_ phase: Phase, step: Int? = nil, totalSteps: Int? = nil) {
        sink?(Event(phase: phase, step: step, totalSteps: totalSteps))
    }
}
