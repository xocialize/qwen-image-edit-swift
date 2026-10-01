// fp32 CPU reference lane for the parity tests (QIE_FP32_CPU / QIF_FP32_CPU).
//
// NOT `Device.setDefault(device: .cpu)`: Swift ops resolve their stream from a default that
// latches on the first MLX op in the process (0.31.6 task-local `_tlDefaultDevice`, 0.32.3
// `Stream.globalStreams`), and setDefault only moves the C++ default afterwards — so any
// earlier op in the process left the "CPU" reference silently on the GPU. When it did
// win (first op in the process) it pinned every later test in the bundle to the CPU.
//
// Compile is disabled inside the lane: the compile cache keys on the C++ default stream,
// which withDefaultDevice leaves on the GPU, so compiled MLXNN activations (silu/gelu)
// first traced on the GPU would replay their GPU tape inside the CPU lane.

import MLX

func withCPUReferenceLane<R>(_ enabled: Bool, _ body: () throws -> R) rethrows -> R {
    guard enabled else { return try body() }
    compile(enable: false)
    defer { compile(enable: true) }
    return try Device.withDefaultDevice(.cpu, body)
}

func withCPUReferenceLane<R>(_ enabled: Bool, _ body: () async throws -> R) async rethrows -> R {
    guard enabled else { return try await body() }
    compile(enable: false)
    defer { compile(enable: true) }
    return try await Device.withDefaultDevice(.cpu, body)
}
