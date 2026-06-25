// One-time conversion: bf16 DiT -> a pre-quantized int4 (mod int8) safetensors. Run this
// once on a big-RAM box (it peaks at the bf16 load); consumers then load the output with
// no bf16 peak. Writes next to the model dir; skips if it already exists.
//
// Run: QIE_CONVERT=1 swift test --filter QuantizedConvertTests

import Foundation
import MLX
import XCTest

@testable import QwenImageEdit

final class QuantizedConvertTests: XCTestCase {
    static let ptDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/"
            + "Qwen-Image-Edit-2511/transformer")
    static let outURL = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/"
            + "qie-2511-dit-int4-mod8.safetensors")
    static let snapshot = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/Qwen-Image-Edit-2511")
    static let encURL = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/qwen-image-edit-models/qie-2511-vl7b-int4.safetensors")

    func testConvert() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_CONVERT"] == "1", "QIE_CONVERT=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.ptDir.path), "missing \(Self.ptDir.path)")
        if FileManager.default.fileExists(atPath: Self.outURL.path) {
            print("[convert] already exists -> \(Self.outURL.path)")
            return
        }
        let config = QwenImageEditWeights.DiTQuantConfig(
            ditBits: 4, modulationBits: 8, groupSize: 64)
        let t = Date()
        try QwenImageEditWeights.saveQuantizedDiT(
            from: Self.ptDir, to: Self.outURL, config: config)
        let bytes = (try? FileManager.default.attributesOfItem(
            atPath: Self.outURL.path)[.size] as? Int) ?? nil
        print("[convert] wrote \(Self.outURL.path) "
            + "(\(bytes.map { String($0 / 1_000_000) } ?? "?") MB) "
            + "in \(String(format: "%.0f", Date().timeIntervalSince(t)))s")
        XCTAssertTrue(FileManager.default.fileExists(atPath: Self.outURL.path))
    }

    func testConvertEncoder() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_CONVERT"] == "1", "QIE_CONVERT=1")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.snapshot.appendingPathComponent("text_encoder").path),
            "missing text_encoder")
        if FileManager.default.fileExists(atPath: Self.encURL.path) {
            print("[convert] encoder already exists -> \(Self.encURL.path)")
            return
        }
        let t = Date()
        try QwenVLPromptEncoder.saveQuantizedTextModel(
            snapshot: Self.snapshot, to: Self.encURL, bits: 4)
        let bytes = (try? FileManager.default.attributesOfItem(
            atPath: Self.encURL.path)[.size] as? Int) ?? nil
        print("[convert] wrote \(Self.encURL.path) "
            + "(\(bytes.map { String($0 / 1_000_000) } ?? "?") MB) "
            + "in \(String(format: "%.0f", Date().timeIntervalSince(t)))s")
        XCTAssertTrue(FileManager.default.fileExists(atPath: Self.encURL.path))
    }
}
