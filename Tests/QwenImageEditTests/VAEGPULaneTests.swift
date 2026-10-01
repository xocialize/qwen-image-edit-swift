// Two-lane VAE gate for the Winograd-free upsampler route: the decode on the GPU lane (route
// on / raw Winograd conv2d) and the CPU lane, fp32 and bf16, against the fp32 references.
//
// Golden: qwen-image-flash-goldens/vae_decode.safetensors — diffusers 0.37.1 fp32 CPU decode at
// 256² (nvidia/Qwen-Image-Flash ships the byte-identical Qwen-Image VAE, md5 8bea68f0…). Its
// `latent_in` is the pipeline's NORMALIZED packed noise (capture_flash_goldens.py decodes
// latent·std + mean), so it is operator parity, not real content. Real content: a DIV2K photo,
// center-cropped to 1024², encoded on the CPU lane (fp32); every decode of that latent is compared
// with the CPU-lane fp32 decode. `testDecodeTiming1024` times the GPU lane alone (no CPU-lane
// buffers in the process).
//
// Run: QIE_PARITY=1 swift test -c release -Xswiftc -enable-testing --filter VAEGPULaneTests
// Overrides: QIE_VAE_DIR, QIE_VAE_GOLDEN, QIE_REAL_IMAGE.

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class VAEGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let vaeDir = URL(
        fileURLWithPath: env["QIE_VAE_DIR"]
            ?? "/Volumes/DEV_ARCHIVE/models/Qwen/Qwen-Image-Edit-2511/vae")
    static let golden = URL(
        fileURLWithPath: env["QIE_VAE_GOLDEN"]
            ?? "/Volumes/DEV_ARCHIVE/models/nvidia/qwen-image-flash-goldens/vae_decode.safetensors")
    static let realImage = URL(
        fileURLWithPath: env["QIE_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")

    struct Stats: CustomStringConvertible {
        let relL2: Float, maxAbs: Float, psnr: Float
        var description: String {
            String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", relL2, maxAbs, psnr)
        }
    }

    /// relL2 / maxAbs / PSNR over [-1, 1] images (peak-to-peak 2).
    static func stats(_ a: MLXArray, _ ref: MLXArray) -> Stats {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        let mse = mean(d * d)
        eval(rel, mx, mse)
        let psnr = 10 * log10(4 / max(mse.item(Float.self), 1e-30))
        return Stats(relL2: rel.item(Float.self), maxAbs: mx.item(Float.self), psnr: psnr)
    }

    /// Compile is off inside the lane: mlx's compile cache keys on the C++ default stream, which
    /// withDefaultDevice leaves on the GPU, so MLXNN's compiled silu traced on the GPU by an earlier
    /// test would replay on the GPU here. Its command buffers then wait on CPU-stream progress and
    /// trip the 5 s GPU watchdog (kIOGPUCommandBufferCallbackErrorTimeout); traced here first, the
    /// GPU lane would run silu on the CPU instead.
    static func onCPU(_ f: () -> MLXArray) -> MLXArray {
        compile(enable: false)
        defer { compile(enable: true) }
        return Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = f()
            eval(r)
            return r
        }
    }

    /// GPU decode on a conv route; warm-up run, then the mean of `reps` timed runs.
    static func gpuDecode(
        _ vae: QwenImageVAE, _ z: MLXArray, route: QwenVAEConvRoute, reps: Int = 3
    ) -> (MLXArray, Double) {
        let saved = vae.convRoute
        vae.convRoute = route
        defer { vae.convRoute = saved }
        var out = vae.decode(z)
        eval(out)
        let t0 = Date()
        for _ in 0..<reps {
            out = vae.decode(z)
            eval(out)
        }
        return (out, Date().timeIntervalSince(t0) / Double(reps) * 1000)
    }

    /// Center crop to side×side, [0,255] RGB → (1, 3, 1, side, side) in [-1, 1].
    static func loadCrop(_ url: URL, side: Int) throws -> MLXArray {
        let img = try EncoderParityTests.loadRGB(url: url)
        precondition(img.width >= side && img.height >= side, "image smaller than crop")
        let (x0, y0) = ((img.width - side) / 2, (img.height - side) / 2)
        var chw = [Float](repeating: 0, count: 3 * side * side)
        let plane = side * side
        for y in 0..<side {
            for x in 0..<side {
                let p = ((y0 + y) * img.width + (x0 + x)) * 3
                let i = y * side + x
                chw[i] = Float(img.rgb[p]) / 127.5 - 1
                chw[plane + i] = Float(img.rgb[p + 1]) / 127.5 - 1
                chw[2 * plane + i] = Float(img.rgb[p + 2]) / 127.5 - 1
            }
        }
        return MLXArray(chw, [1, 3, 1, side, side])
    }

    func testGoldenDecode256() throws {
        try XCTSkipUnless(Self.env["QIE_PARITY"] == "1", "set QIE_PARITY=1 to run")
        let vae = try QwenImageEditWeights.loadVAE(directory: Self.vaeDir, dtype: .float32)
        let g = try MLX.loadArrays(url: Self.golden)
        eval(Array(g.values))  // read the file before a GPU op waits on it (see loadVAE)
        let z = QwenImageVAE.deNormalize(g["latent_in"]!.asType(.float32))
        let ref = g["decoded"]!

        let cpu = Self.onCPU { vae.decode(z) }
        let (route, _) = Self.gpuDecode(vae, z, route: .conv3d)
        let (raw, _) = Self.gpuDecode(vae, z, route: .winograd)
        // diffusers' _decode clamps to [-1, 1] (the golden includes it; noise overshoots a lot).
        func clamped(_ x: MLXArray) -> MLXArray { clip(x, min: -1, max: 1) }
        let sCPU = Self.stats(clamped(cpu), ref)
        let sRoute = Self.stats(clamped(route), ref)
        let sRaw = Self.stats(clamped(raw), ref)
        print("[golden 256² vs diffusers fp32 CPU (clamped)]")
        print("  CPU lane fp32          \(sCPU)")
        print("  GPU fp32, route        \(sRoute)")
        print("  GPU fp32, raw conv2d   \(sRaw)")
        print("  GPU route vs CPU lane  \(Self.stats(route, cpu))")
        print("  GPU raw   vs CPU lane  \(Self.stats(raw, cpu))")
        XCTAssertLessThan(sCPU.relL2, 1e-5, "CPU lane must reproduce the fp32 golden")
        XCTAssertLessThan(Self.stats(route, cpu).relL2, 1e-4, "GPU route vs CPU lane")
    }

    func testRealPhotoDecode1024() throws {
        try XCTSkipUnless(Self.env["QIE_PARITY"] == "1", "set QIE_PARITY=1 to run")
        let vae32 = try QwenImageEditWeights.loadVAE(directory: Self.vaeDir, dtype: .float32)
        let pixels = try Self.loadCrop(Self.realImage, side: 1024)

        // Encode lane check (the encoder has no Winograd-window conv: stride-2 + conv3d only).
        let latCPU = Self.onCPU { vae32.encode(pixels) }
        let latGPU = vae32.encode(pixels)
        eval(latGPU)
        let sEnc = Self.stats(latGPU, latCPU)
        print("[real 1024² photo: \(Self.realImage.lastPathComponent)]")
        print(String(format: "  encode GPU vs CPU (fp32 latents)  relL2 %.2e  maxAbs %.2e",
                     sEnc.relL2, sEnc.maxAbs))

        let z = QwenImageVAE.deNormalize(latCPU)
        let t0 = Date()
        let ref = Self.onCPU { vae32.decode(z) }
        print(String(format: "  CPU-lane fp32 decode (reference): %.1f s", Date().timeIntervalSince(t0)))
        Memory.clearCache()  // the CPU lane leaves tens of GB pooled; don't time against that

        let (r32, t32) = Self.gpuDecode(vae32, z, route: .conv3d)
        let (w32, tw32) = Self.gpuDecode(vae32, z, route: .winograd)
        let vae16 = try QwenImageEditWeights.loadVAE(directory: Self.vaeDir, dtype: .bfloat16)
        let (r16, t16) = Self.gpuDecode(vae16, z, route: .conv3d)
        let (u16, tu16) = Self.gpuDecode(vae16, z, route: .fp32Winograd)
        let (w16, tw16) = Self.gpuDecode(vae16, z, route: .winograd)
        let s = [
            ("GPU fp32, conv3d route  ", Self.stats(r32, ref), t32),
            ("GPU fp32, raw Winograd  ", Self.stats(w32, ref), tw32),
            ("GPU bf16, conv3d route  ", Self.stats(r16, ref), t16),
            ("GPU bf16, fp32 Winograd ", Self.stats(u16, ref), tu16),
            ("GPU bf16, raw Winograd  ", Self.stats(w16, ref), tw16),
        ]
        print("  decode vs CPU-lane fp32:")
        for (label, st, ms) in s { print(String(format: "  %@ %@  %7.1f ms", label, st.description, ms)) }
        XCTAssertLessThan(s[0].1.relL2, 2e-4, "GPU fp32 route vs CPU lane (the rest is TF32 attention)")
    }

    /// GPU-lane decode time at 1024², route vs raw, interleaved over rounds (no CPU lane here).
    func testDecodeTiming1024() throws {
        try XCTSkipUnless(Self.env["QIE_PARITY"] == "1", "set QIE_PARITY=1 to run")
        let pixels = try Self.loadCrop(Self.realImage, side: 1024)
        for dtype in [DType.float32, .bfloat16] {
            let vae = try QwenImageEditWeights.loadVAE(directory: Self.vaeDir, dtype: dtype)
            let z = QwenImageVAE.deNormalize(vae.encode(pixels.asType(dtype)))
            eval(z)
            var t: [QwenVAEConvRoute: [Double]] = [:]
            let routes: [QwenVAEConvRoute] = dtype == .float32 ? [.conv3d, .winograd] : [.conv3d, .fp32Winograd, .winograd]
            for _ in 0..<3 {
                for r in routes { t[r, default: []].append(Self.gpuDecode(vae, z, route: r).1) }
            }
            func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
            let line = routes.map { r in
                String(format: "%@ %@ ms (median %.0f)", r.rawValue,
                       t[r]!.map { String(format: "%.0f", $0) }.joined(separator: "/"), median(t[r]!))
            }.joined(separator: " | ")
            print("[timing 1024² \(dtype == .float32 ? "fp32" : "bf16")] \(line)")
            Memory.clearCache()
        }
    }
}
