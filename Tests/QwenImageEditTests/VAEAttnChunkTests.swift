// `QwenImageVAE.decodeBounded` vs the v0.9.0 decode (AB-L-0176), GPU lane, fp32.
//
// Above `untiledDecodeMaxTokens` the mid-block attention runs in 4096-row query chunks (its
// (h·w)² scores are 3.0 GB at 1328², 17.2 GB at 2048²) and the up path is halo-tiled. Both are
// exact up to GEMM shape. Per size: max|Δ|, relL2, uint8 samples differing, MLX transient and wall
// time, untiled → bounded. 1024² and the edit path's widest ~1 MP output (576×1888) must take the
// untiled path — byte-identical by construction. QIE_VAECHUNK_TILES=64,96 sweeps the tile.
//
// Run: QIE_VAECHUNK=1 [QIE_VAE_DIR=…/vae] swift test -c release -Xswiftc -enable-testing --filter VAEAttnChunkTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class VAEAttnChunkTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let tiles = (ProcessInfo.processInfo.environment["QIE_VAECHUNK_TILES"] ?? "96")
        .split(separator: ",").compactMap { Int($0) }
    static let vaeDir = URL(
        fileURLWithPath: env["QIE_VAE_DIR"]
            ?? "/Volumes/DEV_ARCHIVE/models/Qwen/Qwen-Image-Edit-2511/vae")

    func testChunkedDecodeMatchesUnchunked() throws {
        try XCTSkipUnless(Self.env["QIE_VAECHUNK"] == "1", "QIE_VAECHUNK=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.vaeDir.path), "missing \(Self.vaeDir.path)")
        let vae = try QwenImageEditWeights.loadVAE(directory: Self.vaeDir, dtype: .float32)
        let u8 = { (x: MLXArray) in round(clip(x / 2 + 0.5, min: 0, max: 1) * 255).asType(.uint8) }

        /// v0.9.0 decode (single-matmul attention, untiled) when `tile` is nil, else `decodeBounded`.
        func run(_ z: MLXArray, tile: Int?) -> (MLXArray, Double, Double) {
            MLX.Memory.clearCache()
            let base = MLX.Memory.activeMemory
            MLX.Memory.peakMemory = 0
            let t0 = Date()
            let out = tile.map { vae.decodeBounded(z, tile: $0) } ?? vae.decode(z)
            eval(out)
            return (out, Double(MLX.Memory.peakMemory - base) / 1e9, Date().timeIntervalSince(t0))
        }

        // (latent h, w) → 1024², 576×1888 (widest edit output), 1328² (Qwen-Image native),
        // 2048², 1440×2560 (uneven tiles and last attention chunk: 57,600 = 14·4096 + 256)
        for (h, w) in [(128, 128), (72, 236), (166, 166), (256, 256), (180, 320)] {
            MLXRandom.seed(7)
            let z = MLXRandom.normal([1, 16, 1, h, w])
            let (ref, refPeak, refTime) = run(z, tile: nil)
            let bounded = h * w > vae.untiledDecodeMaxTokens
            for tile in bounded ? Self.tiles : [96] {
                let (out, outPeak, outTime) = run(z, tile: tile)
                let d = abs(out - ref)
                let maxAbs = d.max().item(Float.self)
                let rel = (sqrt(sum(d * d)) / sqrt(sum(ref * ref))).item(Float.self)
                let differ = (u8(out) .!= u8(ref)).asType(.int32).sum().item(Int.self)
                print(String(
                    format: "[vaechunk] %4d×%-4d (latent %d×%d, %@): max|Δ| %.2e relL2 %.2e · uint8 differing %d of %d · MLX transient %.2f → %.2f GB · %.2f → %.2f s",
                    8 * h, 8 * w, h, w, bounded ? "bounded tile \(tile)" : "untiled", maxAbs, rel, differ,
                    ref.size, refPeak, outPeak, refTime, outTime))
                if bounded {
                    XCTAssertLessThan(rel, 1e-5, "bounded decode drifted at \(8 * h)×\(8 * w)")
                } else {
                    XCTAssertEqual(maxAbs, 0, "≤ 1024-bucket decode must be byte-identical")
                }
            }
        }
    }
}
