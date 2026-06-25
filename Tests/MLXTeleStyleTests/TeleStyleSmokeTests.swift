// Phase 3 conformance smoke for the REWIRED TeleStyle: base 2511 + style + Lightning
// LoRAs applied at runtime (rank-stacked), driven through the ModelPackage interface.
// Confirms the bf16-fuse path is gone and a 4-step style transfer returns a valid PNG.
//
// Run: QIE_TELESTYLE_PKG=1 swift test --filter TeleStyleSmokeTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXTeleStyle

final class TeleStyleSmokeTests: XCTestCase {
    func testStyleTransfer() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TELESTYLE_PKG"] == "1", "QIE_TELESTYLE_PKG=1")

        let config = TeleStyleConfiguration()  // base + style + lightning, 4-step DMD defaults
        for p in [config.basePath, config.styleLoRAPath, config.lightningLoRAPath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }
        let assets = "/Users/dustinnielson/Development/telestyle-work/tele_assets"
        let contentPath = "\(assets)/content_1.webp"
        let stylePath = "\(assets)/style_1.jpg"
        for p in [contentPath, stylePath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "missing \(p)")
        }

        func image(_ path: String) throws -> Image {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let (_, w, h) = try TeleStylePackage.decodeRGB(data)
            return Image(format: .png, data: data, width: w, height: h)
        }
        // image 0 = content, image 1 = style.
        let request = IEditRequest(
            images: [try image(contentPath), try image(stylePath)],
            prompt: "Apply the style of the second image to the first.",
            steps: 4, guidanceScale: 1.0, seed: 42, mode: styleTransferMode)

        let pkg = TeleStylePackage(configuration: config)
        try await pkg.load()  // loads base + rank-stacks style+Lightning at runtime
        let response = try await pkg.run(request)

        guard let edit = response as? IEditResponse else {
            return XCTFail("expected IEditResponse, got \(type(of: response))")
        }
        XCTAssertEqual(edit.image.format, .png)
        XCTAssertGreaterThan(edit.image.data.count, 1000)
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qie-telestyle-runtime-smoke.png")
        try edit.image.data.write(to: out)
        print("[telestyle-pkg] \(edit.image.width ?? 0)x\(edit.image.height ?? 0), "
            + "\(edit.image.data.count) bytes -> \(out.path)")
    }
}
