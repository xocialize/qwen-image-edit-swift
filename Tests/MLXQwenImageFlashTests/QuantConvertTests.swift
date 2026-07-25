// One-time conversion: the bf16 Flash snapshot -> a pre-quantized int4 snapshot.
//
// Runs at the bf16 peak (~52 GB), so it happens ONCE on a big-RAM box; consumers then load
// the output with no bf16 peak at all (QuantizedDiT.swift's load trick: build the module,
// install the QuantizedLinear structure with the same filter, load the int4 weights straight
// in). That asymmetry is the whole point — a 36 GB Mac cannot load 41 GB of bf16 in order to
// quantize it, so a load-time-quantization tier would not run on the devices this tier is for.
//
// The T2I path never builds the ViT (loadTextOnly), so the int4 text encoder carries the
// language model only — no `visual.*`.
//
// Run: QIF_CONVERT=1 swift test --filter FlashQuantConvertTests

import Foundation
import MLX
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashQuantConvertTests: XCTestCase {
    static let source = URL(fileURLWithPath: FlashPackageTests.snapshot)
    static let outDir = URL(
        fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/nvidia/Qwen-Image-Flash-4bit")

    func testConvertDiT() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_CONVERT"] == "1", "QIF_CONVERT=1")
        let ptDir = Self.source.appendingPathComponent("transformer")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: ptDir.path), "missing \(ptDir.path)")

        let out = Self.outDir.appendingPathComponent("transformer/model-int4.safetensors")
        try FileManager.default.createDirectory(
            at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: out.path) {
            print("[convert] DiT already exists -> \(out.path)")
            return
        }
        // int4 attn+FFN, int8 modulation: the conditioning-critical img_mod/txt_mod linears
        // are grainy at int4, and they are a rounding error in size next to the blocks.
        let config = QwenImageEditWeights.DiTQuantConfig(
            ditBits: 4, modulationBits: 8, groupSize: 64)
        let t = Date()
        try QwenImageEditWeights.saveQuantizedDiT(from: ptDir, to: out, config: config)
        let mb = ((try? FileManager.default.attributesOfItem(atPath: out.path)[.size]) as? Int
            ?? 0) / 1_000_000
        print("[convert] DiT int4/mod8 -> \(out.path) (\(mb) MB) in "
            + String(format: "%.0f s", Date().timeIntervalSince(t)))
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path))
    }

    func testConvertTextEncoder() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_CONVERT"] == "1", "QIF_CONVERT=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.source.appendingPathComponent("text_encoder").path),
            "missing text_encoder")

        let out = Self.outDir.appendingPathComponent("text_encoder/model-int4.safetensors")
        try FileManager.default.createDirectory(
            at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: out.path) {
            print("[convert] encoder already exists -> \(out.path)")
            return
        }
        let t = Date()
        try QwenVLPromptEncoder.saveQuantizedTextModel(
            snapshot: Self.source, to: out, bits: 4, groupSize: 64)
        let mb = ((try? FileManager.default.attributesOfItem(atPath: out.path)[.size]) as? Int
            ?? 0) / 1_000_000
        print("[convert] VL-7B text int4 -> \(out.path) (\(mb) MB) in "
            + String(format: "%.0f s", Date().timeIntervalSince(t)))
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path))
    }

    /// Copy the small companions so the int4 directory is a self-contained snapshot: configs,
    /// tokenizer (incl. the converted tokenizer.json), scheduler, and the VAE (which stays
    /// full precision — it is 0.25 GB and precision-sensitive).
    func testAssembleSnapshot() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIF_CONVERT"] == "1", "QIF_CONVERT=1")
        let fm = FileManager.default
        try fm.createDirectory(at: Self.outDir, withIntermediateDirectories: true)
        for item in [
            "vae", "tokenizer", "scheduler", "model_index.json", "LICENSE",
            "text_encoder/config.json", "transformer/config.json",
        ] {
            let src = Self.source.appendingPathComponent(item)
            let dst = Self.outDir.appendingPathComponent(item)
            guard fm.fileExists(atPath: src.path) else { continue }
            if fm.fileExists(atPath: dst.path) { continue }
            try fm.createDirectory(
                at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: src, to: dst)
            print("[assemble] \(item)")
        }
        XCTAssertTrue(
            fm.fileExists(atPath: Self.outDir.appendingPathComponent("vae/config.json").path))
    }
}
