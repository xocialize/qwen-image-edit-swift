// Phase 2 conformance smoke: drive the fast tier through the engine-facing ModelPackage
// interface (init -> load -> run(IEditRequest) -> IEditResponse). Confirms the Lightning
// LoRA is applied at load and a 4-step DMD edit returns a valid PNG.
//
// Run: QIE_TURBO_PKG=1 swift test --filter TurboPackageSmokeTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXQwenImageEditTurbo

final class TurboPackageSmokeTests: XCTestCase {
    func testLoadAndEdit() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_TURBO_PKG"] == "1", "QIE_TURBO_PKG=1")

        let config = QwenImageEditTurboConfiguration()  // base + Lightning + 4-step DMD defaults
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: config.loraPath), "missing \(config.loraPath)")
        let inputPath = "/Users/dustinnielson/Desktop/lens-t2i-package.png"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: inputPath), "missing \(inputPath)")

        let data = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        let (_, w, h) = try QwenImageEditTurboPackage.decodeRGB(data)
        let request = IEditRequest(
            images: [Image(format: .png, data: data, width: w, height: h)],
            prompt: "Change the fox's fur color to snow white. Keep the pose, background and "
                + "lighting unchanged.",
            steps: 4, guidanceScale: 1.0, seed: 42, mode: turboMode)

        let pkg = QwenImageEditTurboPackage(configuration: config)
        try await pkg.load()
        let response = try await pkg.run(request)

        guard let edit = response as? IEditResponse else {
            return XCTFail("expected IEditResponse, got \(type(of: response))")
        }
        XCTAssertEqual(edit.image.format, .png)
        XCTAssertGreaterThan(edit.image.data.count, 1000, "PNG suspiciously small")
        XCTAssertEqual(edit.image.width, 1024)
        XCTAssertEqual(edit.image.height, 1024)

        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qie-turbo-pkg-smoke.png")
        try edit.image.data.write(to: out)
        print("[turbo-pkg] wrote \(edit.image.data.count) bytes -> \(out.path)")
    }
}
