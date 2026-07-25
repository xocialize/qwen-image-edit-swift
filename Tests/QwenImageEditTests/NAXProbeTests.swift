// NAX split-K GEMM probe + the invariant that protects this DiT from it (mlx#3797, fixed
// upstream by #3810).
//
// On mlx-swift ≤0.31.6 with MLX_METAL_JIT=ON, a half-precision matmul in the window
//   M·N ≥ 2048² , K ≥ 10240 , K ≥ 3·max(M,N)
// is mis-instantiated with the output dtype and returns garbage on M5-class GPUs.
//
// For this DiT the img-stream FFN down-projection is K = 4·3072 = 12288, N = 3072, so the
// qualifying window is 1366 ≤ M ≤ 4096 image tokens — output grids from ~592² up to exactly
// 1024². The EDIT path never entered it (target+conditioning tokens put M at 8192, where
// K ≥ 3·M fails); the T2I path lands at M = 4096 at the model's tested 1024² size, dead on
// the upper edge. `QwenFeedForward.downProjected` row-chunks at ≤896 to stay out of it.
//
// Measured on this box / mlx-swift 0.31.6 (raw GEMM, QIE_NAX_PROBE=1):
//   M=256 cos 0.99999887 · M=1366 cos 0.70700407 · M=2048 cos 0.0 · M=4096 NaN
//
// Removal path: bump the mlx-swift pin, run `QIE_NAX_PROBE=1 swift test --filter NAXProbeTests`;
// when the raw probe PASSES edge-to-edge, delete the chunk in QwenFeedForward.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import QwenImageEdit

final class NAXProbeTests: XCTestCase {

    private static func lcgArray(_ shape: [Int], seed: UInt64 = 0x9E37_79B9_7F4A_7C15)
        -> MLXArray
    {
        var lcg = seed
        let n = shape.reduce(1, *)
        let v = (0..<n).map { _ -> Float in
            lcg = lcg &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int64(bitPattern: lcg >> 11)) / Float(Int64.max >> 11)
        }
        return MLXArray(v, shape)
    }

    private static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).flattened()
        let y = b.asType(.float32).flattened()
        let c = sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)) + 1e-12)
        eval(c)
        return c.item(Float.self)
    }

    /// Raw-GEMM diagnostic at the DiT FFN shape. EXPECTED TO FAIL while the pin is ≤0.31.6 —
    /// it is the removal signal for the row-chunk, not a health check, hence env-gated.
    func testFeedForwardGEMMWindow() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_NAX_PROBE"] == "1",
            "set QIE_NAX_PROBE=1 — diagnostic; fails by design until mlx-swift vendors #3810")
        let (K, N) = (12288, 3072)
        let b = Self.lcgArray([N, K]).asType(.bfloat16)
        // 256 = the parity-golden grid (below the window); 1366 = first qualifying M;
        // 4096 = 1024² T2I (last qualifying M — K ≥ 3·M fails at 4097); 8192 = the edit
        // path's target+cond sequence (above the window).
        for m in [256, 1366, 2048, 4096, 8192] {
            let a = Self.lcgArray([m, K], seed: UInt64(m) &* 2_654_435_761).asType(.bfloat16)
            let y = matmul(a, b.T)
            let ref = matmul(a.asType(.float32), b.asType(.float32).T)
            eval(y, ref)
            let cos = Self.cosine(y, ref)
            let maxAbs = abs(y.asType(.float32) - ref).max().item(Float.self)
            print(String(format: "  M=%d K=%d N=%d bf16: cos %.8f max_abs %.3e", m, K, N, cos, maxAbs))
            // STRICT on purpose: corruption just past the boundary is subtle — cos ~0.998
            // passes a loose ≥0.99 check and is still garbage (the Boogu lesson).
            XCTAssertGreaterThan(cos, 0.999, "M=\(m): NAX split-K corruption (mlx#3797)")
            XCTAssertTrue(maxAbs.isFinite && maxAbs < 100, "M=\(m): max_abs \(maxAbs)")
        }
    }

    /// The invariant that keeps 1024² T2I renderable on the current pin: the row-chunked
    /// down-projection must match the fp32 reference at M = 4096, where the unchunked bf16
    /// GEMM returns NaN. Random weights — this tests the kernel path, not the model.
    func testChunkedFeedForwardIsExactAtProductionGrid() {
        let (dim, hidden) = (3072, 12288)
        let ff = QwenFeedForward(dim: dim, hiddenDim: hidden)
        ff.update(parameters: ModuleParameters.unflattened([
            "proj_in.weight": Self.lcgArray([hidden, dim], seed: 11).asType(.bfloat16) * 0.02,
            "proj_in.bias": MLXArray.zeros([hidden]).asType(.bfloat16),
            "proj_out.weight": Self.lcgArray([dim, hidden], seed: 22).asType(.bfloat16) * 0.02,
            "proj_out.bias": MLXArray.zeros([dim]).asType(.bfloat16),
        ]))
        eval(ff)

        let m = 4096  // 1024² T2I image tokens — the NaN case unchunked
        let x = (Self.lcgArray([1, m, dim], seed: 33) * 0.5).asType(.bfloat16)
        let chunked = ff(x)
        eval(chunked)
        XCTAssertTrue(
            chunked.asType(.float32).max().item(Float.self).isFinite,
            "chunked down-projection produced non-finite output at M=\(m)")

        // fp32 reference: the dtype guard routes fp32 straight through, unchunked.
        let ff32 = QwenFeedForward(dim: dim, hiddenDim: hidden)
        ff32.update(parameters: ModuleParameters.unflattened([
            "proj_in.weight": ff.projIn.weight.asType(.float32),
            "proj_in.bias": ff.projIn.bias!.asType(.float32),
            "proj_out.weight": ff.projOut.weight.asType(.float32),
            "proj_out.bias": ff.projOut.bias!.asType(.float32),
        ]))
        eval(ff32)
        let reference = ff32(x.asType(.float32))
        eval(reference)

        let cos = Self.cosine(chunked, reference)
        print(String(format: "  chunked FFN @ M=%d: cos %.8f", m, cos))
        XCTAssertGreaterThan(cos, 0.999, "row-chunked down-projection diverged from fp32")
    }
}
