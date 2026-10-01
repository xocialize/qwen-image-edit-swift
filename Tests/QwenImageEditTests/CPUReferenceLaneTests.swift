// Regression guard for the fp32 CPU reference lane (CPUReferenceLane.swift) the parity tests
// use under QIE_FP32_CPU / QIF_FP32_CPU.
//
// Detectors (no timing, so a busy shared GPU can't fool them):
// - MLXLinalg.inv throws at graph construction when its resolved stream is the GPU, so
//   `inv` on the default stream succeeds iff Swift ops are routed to the CPU.
// - fp32 matmul differs bitwise between CPU and GPU (accumulation order), so a result that
//   equals the GPU bits was computed on the GPU.
//
// Measured 2026-10-01 on mlx-swift 0.31.6 and 0.32.3: setDefault(.cpu) after any MLX op left
// ops on the GPU; inside withDefaultDevice(.cpu) a compiled fn or MLXNN silu first traced on
// the GPU replayed its GPU tape; compile(enable: false) gave true CPU results. Before the lane,
// T2I dit_step0 "fp32 CPU" read 0.99950373 (GPU, gate fails) when an earlier test class ran
// MLX ops in the same process; with it, 1.0000002 in every ordering.
//
// The testCanary* cases pin the upstream behaviour the lane works around. If one fails after
// an mlx-swift bump, upstream changed: re-evaluate the workaround (e.g. drop the compile
// toggle if the compile cache starts keying on the Swift default device).

import MLX
import MLXNN
import XCTest

final class CPUReferenceLaneTests: XCTestCase {

    /// "cpu" if the current default stream accepts a CPU-only op, else "gpu".
    static func lane() -> String {
        let x = MLXArray.eye(4) * 2
        do {
            let y = try withError { MLXLinalg.inv(x) }
            eval(y)
            return "cpu"
        } catch {
            return "gpu"
        }
    }

    static func same(_ a: MLXArray, _ b: MLXArray) -> Bool { a.arrayEqual(b).item(Bool.self) }

    /// silu(a·a) with no compile anywhere — the CPU ground truth when run in a CPU scope.
    static func hand(_ a: MLXArray) -> MLXArray {
        let m = matmul(a, a)
        return m * sigmoid(m)
    }

    // MARK: - The lane

    func testLaneRoutesToCPUAfterGPUWork() {
        eval(MLXArray(Float(1)) + 1)  // latch the process default (GPU) first
        XCTAssertEqual(withCPUReferenceLane(true) { Self.lane() }, "cpu")
        XCTAssertEqual(withCPUReferenceLane(false) { Self.lane() }, "gpu")
        XCTAssertEqual(Self.lane(), "gpu", "the lane must not leak past its scope")
    }

    func testLaneRoutesToCPUAcrossAwait() async {
        eval(MLXArray(Float(1)) + 1)
        let lane = await withCPUReferenceLane(true) { () async -> String in
            await Task.yield()
            return Self.lane()
        }
        XCTAssertEqual(lane, "cpu")
        XCTAssertEqual(Self.lane(), "gpu")
    }

    func testLaneGivesTrueCPUResultsForGPUTracedCompiles() {
        let x = MLXRandom.normal([512, 512], key: MLXRandom.key(0))
        let m = MLXRandom.normal([1024, 1024], key: MLXRandom.key(1)) * 8
        let f = compile { (a: MLXArray) -> MLXArray in silu(matmul(a, a)) }
        let gpuHand = Self.hand(x)
        eval(f(x), silu(m), gpuHand)  // trace f and MLXNN silu on the GPU

        let (cpuHand, cpuSilu) = Device.withDefaultDevice(.cpu) { () -> (MLXArray, MLXArray) in
            let h = Self.hand(x)
            let s = m * sigmoid(m)
            eval(h, s)
            return (h, s)
        }
        XCTAssertFalse(Self.same(cpuHand, gpuHand), "detector needs CPU and GPU matmul to differ")

        let (laneF, laneSilu) = withCPUReferenceLane(true) { () -> (MLXArray, MLXArray) in
            let a = f(x)
            let b = silu(m)
            eval(a, b)
            return (a, b)
        }
        XCTAssertTrue(Self.same(laneF, cpuHand), "compiled fn replayed its GPU trace in the lane")
        XCTAssertTrue(Self.same(laneSilu, cpuSilu), "MLXNN silu replayed its GPU trace in the lane")
    }

    // MARK: - Canaries: upstream behaviour the lane works around

    /// Why the lane is not setDefault: after the first op, setDefault only moves the C++
    /// default (0.31.6 latches the task-local default, 0.32.3 Stream.globalStreams).
    @available(*, deprecated, message: "exercises the deprecated Device.setDefault on purpose")
    func testCanarySetDefaultAfterFirstOpLeavesOpsOnGPU() {
        eval(MLXArray(Float(1)) + 1)
        Device.setDefault(device: Device(.cpu))
        defer { Device.setDefault(device: Device(.gpu)) }
        XCTAssertEqual(Self.lane(), "gpu")
    }

    /// Why the lane disables compile: the cache keys on the C++ default stream, which
    /// withDefaultDevice leaves on the GPU, so GPU-traced tapes replay in a CPU scope.
    func testCanaryCompileCacheReplaysGPUTraceInCPUScope() {
        let x = MLXRandom.normal([512, 512], key: MLXRandom.key(2))
        let m = MLXRandom.normal([1024, 1024], key: MLXRandom.key(3)) * 8
        let f = compile { (a: MLXArray) -> MLXArray in silu(matmul(a, a)) }
        let gpuF = f(x)
        let gpuSilu = silu(m)  // top-level call: MLXNN's cache entry traced on the GPU
        eval(gpuF, gpuSilu)

        let fresh = compile(shapeless: true) { (a: MLXArray) -> MLXArray in a * sigmoid(a) }
        let (cpuF, cpuSilu, freshSilu) = Device.withDefaultDevice(.cpu) {
            () -> (MLXArray, MLXArray, MLXArray) in
            let a = f(x)
            let b = silu(m)
            let c = fresh(m)  // first traced here: a genuine CPU compiled kernel
            eval(a, b, c)
            return (a, b, c)
        }
        XCTAssertFalse(Self.same(freshSilu, gpuSilu), "detector needs CPU and GPU silu to differ")
        XCTAssertTrue(Self.same(cpuF, gpuF), "compiled fn no longer replays the GPU tape")
        XCTAssertTrue(Self.same(cpuSilu, gpuSilu), "MLXNN silu no longer replays the GPU tape")
    }

    /// Same cache keying, reverse direction and fully deterministic: a CPU-only op traced in a
    /// CPU scope replays on the CPU when called on the GPU default (a re-trace would throw).
    func testCanaryCompileCacheIgnoresSwiftDevice() {
        eval(MLXArray(Float(1)) + 1)
        let g = compile { (a: MLXArray) -> MLXArray in MLXLinalg.inv(a) }
        let x = MLXArray.eye(4) * 2
        Device.withDefaultDevice(.cpu) { eval(g(x)) }
        do {
            let y = try withError { g(x) }  // eval outside: a failed op yields no array
            eval(y)
        } catch {
            XCTFail("cache now keys on the Swift device: \(error)")
        }
    }
}
