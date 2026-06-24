// MLXTeleStyle conformance + e2e through the engine `imageEdit` contract.
//
// Manifest checks run always (C-gate). The e2e style-transfer run is gated on
// TELESTYLE_DEMO=1 (loads the fused 60 GB snapshot).

import Foundation
import MLXToolKit
import XCTest

@testable import MLXTeleStyle

final class TeleStylePackageTests: XCTestCase {

    func testManifestConformance() {
        let m = TeleStylePackage.manifest
        // Surface = imageEdit with the styleTransfer mode declared (C4/C11).
        XCTAssertEqual(m.surfaces.count, 1)
        let s = m.surfaces[0]
        XCTAssertEqual(s.capability, .imageEdit)
        XCTAssertTrue(s.supportedModes.contains(styleTransferMode))
        // License gate: Apache weights + MIT port code (C7/C8).
        XCTAssertEqual(m.license.weightLicense, .apache2)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.provenance.sourceRepo, "Tele-AI/TeleStyleV2")
        // Measured footprint present (C-gate: no placeholder).
        XCTAssertFalse(m.requirements.footprints.isEmpty)
        XCTAssertGreaterThan(m.requirements.footprints[0].residentBytes, 0)
    }

    func testRegistrationOneLiner() {
        // Author one-liner the engine registers — must resolve without throwing.
        _ = TeleStylePackage.registration
    }

    func testStyleTransferE2E() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["TELESTYLE_DEMO"] == "1", "TELESTYLE_DEMO=1")
        let assets = URL(
            fileURLWithPath: "/Users/dustinnielson/Development/telestyle-work/tele_assets")
        let content = try Data(contentsOf: assets.appendingPathComponent("content_1.webp"))
        let style = try Data(contentsOf: assets.appendingPathComponent("style_1.jpg"))

        let pkg = TeleStylePackage(configuration: .init())
        try await pkg.load()
        let req = IEditRequest(
            images: [
                Image(format: .png, data: content, width: 0, height: 0),
                Image(format: .png, data: style, width: 0, height: 0),
            ],
            prompt: "Style Transfer the style of Figure 2 to Figure 1, and keep the "
                + "content and characteristics of Figure 1.",
            seed: 123,
            mode: styleTransferMode)
        let resp = try await pkg.run(req) as! IEditResponse
        XCTAssertGreaterThan(resp.image.data.count, 10_000)
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/telestyle-package-e2e.png")
        try resp.image.data.write(to: out)
        print("saved \(out.path)  (\(resp.image.width)x\(resp.image.height))")
    }
}
