// Quality eval: the rewired (runtime multi-LoRA) TeleStyle at increasing step counts,
// to set a sensible default — the 4-step DMD output is softer than the multi-step
// reference, so check whether crispness is purely a step-count tradeoff. LoRAs are
// applied at load (not per step), so one load covers the whole sweep.
//
// Run: QIE_TELESTYLE_STEPS=1 swift test --filter TeleStyleStepsEvalTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXTeleStyle

final class TeleStyleStepsEvalTests: XCTestCase {
    func testStepsSweep() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TELESTYLE_STEPS"] == "1", "QIE_TELESTYLE_STEPS=1")
        let config = TeleStyleConfiguration()
        for p in [config.basePath, config.styleLoRAPath, config.lightningLoRAPath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }
        let assets = "/Users/dustinnielson/Development/telestyle-work/tele_assets"
        func image(_ path: String) throws -> Image {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let (_, w, h) = try TeleStylePackage.decodeRGB(data)
            return Image(format: .png, data: data, width: w, height: h)
        }
        let images = [try image("\(assets)/content_1.webp"), try image("\(assets)/style_1.jpg")]

        let pkg = TeleStylePackage(configuration: config)
        try await pkg.load()
        let desktop = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop")
        for steps in [4, 8, 16] {
            let t = Date()
            let req = IEditRequest(
                images: images, prompt: "Apply the style of the second image to the first.",
                steps: steps, guidanceScale: 1.0, seed: 42, mode: styleTransferMode)
            let resp = try await pkg.run(req)
            let edit = resp as! IEditResponse
            let out = desktop.appendingPathComponent("qie-telestyle-s\(steps).png")
            try edit.image.data.write(to: out)
            print("[telestyle-steps] \(steps) steps in "
                + "\(String(format: "%.0f", Date().timeIntervalSince(t)))s -> \(out.path)")
        }
    }
}
