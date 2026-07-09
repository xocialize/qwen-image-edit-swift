// FR2 live quality/speed gate: same-seed 20-step CFG-4 edits with the step-residual
// cache off / conservative / aggressive. Score the cached outputs against the uncached
// one with SSIMULACRA2 (≥ 85 = "excellent", the default-on bar; presets lore 90/80/70):
//
//   QIE_STEPCACHE_EVAL=1 [QIE_STEPCACHE_OUT=dir] swift test --filter StepCacheEvalTests
//   ssimulacra2 <out>/stepcache-off.png <out>/stepcache-{conservative,aggressive}.png
//
// Also live-exercises the FR6b prompt-embedding memo: all three runs share (images,
// prompt, negative), so runs 2 and 3 must skip the per-request VL-7B load + encode —
// visible as a missing "encode vl" span (MLX_PROFILE=1) and a faster wall clock.

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class StepCacheEvalTests: XCTestCase {
    func testStepCacheAB() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_STEPCACHE_EVAL"] == "1",
            "QIE_STEPCACHE_EVAL=1")

        // No historical golden needed: the same-run UNCACHED output is the SSIMULACRA2
        // reference for the cached variants.
        let env = ProcessInfo.processInfo.environment
        let modelDir = URL(
            fileURLWithPath: env["QIE_MODEL_DIR"]
                ?? "/Volumes/DEV_ARCHIVE/models/Qwen/Qwen-Image-Edit-2511")
        let imagePath = env["QIE_STEPCACHE_IMAGE"]
            ?? "/Users/dustinnielson/Development/telestyle-work/tele_assets/content_1.webp"
        let prompt = env["QIE_STEPCACHE_PROMPT"]
            ?? "Change the season to a snowy winter day while keeping the subject, "
            + "composition and lighting direction unchanged."
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("transformer").path),
            "missing snapshot at \(modelDir.path)")
        let image = try EncoderParityTests.loadRGB(url: URL(fileURLWithPath: imagePath))
        let outDir = URL(
            fileURLWithPath: env["QIE_STEPCACHE_OUT"]
                ?? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Desktop").path)
        try FileManager.default.createDirectory(
            at: outDir, withIntermediateDirectories: true)

        // Production residency: per-request encoder load + evict. Runs 2/3 hit the FR6b
        // prompt memo, so only run 1 pays the encoder load at all.
        let transformer = try QwenImageEditWeights.loadDiTFromPT(
            directory: modelDir.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImageEditWeights.loadVAE(
            directory: modelDir.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImageEditGenerator(
            encoderProvider: { try await QwenVLPromptEncoder.load(snapshot: modelDir) },
            transformer: transformer, vae: vae)
        generator.warmup()

        // QIE_STEPCACHE_MODES=conservative,aggressive re-runs a subset (the seeded
        // uncached reference is deterministic — no need to regenerate it after
        // cache-metric changes that leave the off path untouched).
        let modes = env["QIE_STEPCACHE_MODES"]
            .map { $0.split(separator: ",").compactMap { StepCacheMode(rawValue: String($0)) } }
            ?? StepCacheMode.allCases
        for mode in modes {
            let start = Date()
            let (pixels, w, h) = try await generator.generate(
                image: image,
                prompt: prompt,
                negativePrompt: " ",
                steps: 20,
                trueCFGScale: 4.0,
                seed: 42,
                stepCache: mode,
                progress: { _, _ in })
            let secs = Date().timeIntervalSince(start)
            let skips = generator.lastStepCacheSkips.map { "pos \($0.pos)/20 neg \($0.neg)/20" }
                ?? "cache off"
            let out = outDir.appendingPathComponent("stepcache-\(mode.rawValue).png")
            try GenerateDemoTests.writePNG(pixels: pixels, width: w, height: h, to: out)
            print(
                "[stepcache-eval] \(mode.rawValue): \(String(format: "%.1f", secs))s, "
                    + "skipped \(skips) -> \(out.path)")
        }
    }
}
