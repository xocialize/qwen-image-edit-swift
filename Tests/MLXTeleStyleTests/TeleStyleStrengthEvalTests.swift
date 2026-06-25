// Strength experiment: at a FIXED 4 steps, does the Lightning DMD LoRA actually help the
// hard global restyle, and does its strength matter? Sweeps lightningStrength while
// holding style at 1.0; reloads per value (LoRAs bake at load). Compare against the
// existing qie-telestyle-s4 (strength 1) and qie-telestyle-s16 (strength 1) renders.
//   strength 0 = style-only control (lightning scale 0 -> no contribution)
//
// Run: QIE_TELESTYLE_STRENGTH=1 swift test --filter TeleStyleStrengthEvalTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXTeleStyle

final class TeleStyleStrengthEvalTests: XCTestCase {
    func testLightningStrengthAt4Steps() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TELESTYLE_STRENGTH"] == "1",
            "QIE_TELESTYLE_STRENGTH=1")
        let probe = TeleStyleConfiguration()
        for p in [probe.basePath, probe.styleLoRAPath, probe.lightningLoRAPath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }
        let assets = "/Users/dustinnielson/Development/telestyle-work/tele_assets"
        func image(_ path: String) throws -> Image {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let (_, w, h) = try TeleStylePackage.decodeRGB(data)
            return Image(format: .png, data: data, width: w, height: h)
        }
        let images = [try image("\(assets)/content_1.webp"), try image("\(assets)/style_1.jpg")]
        let desktop = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop")

        for lightning in [Float(0.0), 4.0] {
            let config = TeleStyleConfiguration(lightningStrength: lightning)
            let pkg = TeleStylePackage(configuration: config)
            try await pkg.load()  // base + style@1 + lightning@<lightning>, rank-stacked
            let t = Date()
            let req = IEditRequest(
                images: images, prompt: "Apply the style of the second image to the first.",
                steps: 4, guidanceScale: 1.0, seed: 42, mode: styleTransferMode)
            let edit = try await pkg.run(req) as! IEditResponse
            let tag = String(format: "%.0f", lightning)
            let out = desktop.appendingPathComponent("qie-telestyle-s4-light\(tag).png")
            try edit.image.data.write(to: out)
            await pkg.unload()
            print("[telestyle-strength] light=\(lightning) @4steps in "
                + "\(String(format: "%.0f", Date().timeIntervalSince(t)))s -> \(out.path)")
        }
    }
}
