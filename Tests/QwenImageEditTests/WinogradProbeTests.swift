// Weight-free probe: mlx's Winograd conv2d path at this VAE's upsampler shapes, raw vs the
// WinogradFreeConv2d route (Sources/QwenImageEdit/WinogradFreeConv2d.swift), both against the
// CPU conv (exact-class).
//
// Measured 2026-09-24 on the M5 Max, mlx-swift 0.31.6 (GPU vs CPU, fp32 relL2):
//   raw conv2d 384→192 @128² 6.4e-3 · 192→96 @256² 6.4e-3 (Winograd) — route ~1e-6.
// Removal signal: when a new mlx-swift pin reports raw conv2d exact-class here, drop the route.
//
// Run: swift test --filter WinogradProbeTests

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import QwenImageEdit

final class WinogradProbeTests: XCTestCase {

    static func relL2(_ a: MLXArray, _ b: MLXArray) -> (rel: Float, maxAbs: Float) {
        let d = a.asType(.float32) - b.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(b.asType(.float32))))
        let mx = abs(d).max()
        eval(rel, mx)
        return (rel.item(Float.self), mx.item(Float.self))
    }

    func testUpsamplerConvShapes() throws {
        for (c, o, s) in [(384, 192, 128), (192, 96, 256)] {
            MLXRandom.seed(UInt64(c))
            let conv = WinogradFreeConv2d(
                inputChannels: c, outputChannels: o, kernelSize: 3, padding: 1)
            let x = MLXRandom.normal([1, s, s, c])
            XCTAssertTrue(
                WinogradFreeConv2d.takesWinograd(
                    input: x, weight: conv.weight, stride: conv.stride,
                    dilation: conv.dilation, groups: conv.groups),
                "\(c)→\(o) @\(s)² should be inside the Winograd predicate")
            let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
                let r = conv2d(x, conv.weight, stride: 1, padding: 1) + conv.bias!
                eval(r)
                return r
            }
            conv.enabled = false
            let raw = conv(x)
            conv.enabled = true
            let routed = conv(x)
            eval(raw, routed)
            let r0 = Self.relL2(raw, ref)
            let r1 = Self.relL2(routed, ref)
            print(
                String(
                    format: "  %d→%d @%d²: raw conv2d relL2 %.2e (max %.1e) %@ | route relL2 %.2e",
                    c, o, s, r0.rel, r0.maxAbs,
                    r0.rel < 1e-5 ? "EXACT — route removable" : "lossy", r1.rel))
            XCTAssertLessThan(r1.rel, 1e-5, "conv3d route must be exact-class")
        }
    }
}
