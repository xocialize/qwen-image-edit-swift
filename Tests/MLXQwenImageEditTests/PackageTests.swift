// Engine-conformance smoke: manifest + load -> run(IEditRequest) -> PNG -> unload.
//
// Run: QIE_PKG=1 swift test --filter PackageTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXQwenImageEdit

final class PackageTests: XCTestCase {
    func testManifest() {
        let m = QwenImageEditPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .imageEdit)
        XCTAssertEqual(m.license.weightLicense, .apache2)
    }

    func testLoadRunUnload() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QIE_PKG"] == "1", "QIE_PKG=1")

        let package = QwenImageEditPackage(configuration: .init())
        try await package.load()

        let foxData = try Data(
            contentsOf: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Desktop/lens-t2i-package.png"))
        let request = IEditRequest(
            images: [Image(format: .png, data: foxData)],
            prompt: "Give the fox a small red scarf around its neck. Keep everything else unchanged.",
            steps: 8,
            guidanceScale: 4.0,
            seed: 7)
        let start = Date()
        let response = try await package.run(request)
        guard let edit = response as? IEditResponse else {
            return XCTFail("wrong response type")
        }
        print("package edit: \(edit.image.width ?? 0)x\(edit.image.height ?? 0) in \(Date().timeIntervalSince(start))s")
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qwen-edit-package-demo.png")
        try edit.image.data.write(to: out)
        print("saved \(out.path)")

        await package.unload()
    }
}
