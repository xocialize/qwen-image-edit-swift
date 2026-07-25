// Engine-conformance for the Qwen-Image-Flash textToImage package: manifest + MAT gate
// offline; load -> run(T2IRequest) -> PNG -> unload behind QIF_PKG=1.
//
// Run: swift test --filter MLXQwenImageFlashTests
//      QIF_PKG=1 swift test --filter PackageTests

import Foundation
import MLX
import MLXServeConformance
import MLXToolKit
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashPackageTests: XCTestCase {

    static let snapshot = "/Volumes/DEV_ARCHIVE/models/nvidia/Qwen-Image-Flash"

    func testManifest() {
        let m = QwenImageFlashPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .textToImage)
        // C7: NVIDIA Open Model License (allowlisted permissive); C8: MIT port code.
        XCTAssertEqual(m.license.weightLicense, .nvidiaOpenModel)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertTrue(m.license.weightLicense.isPermissive)
        // Split footprint, not a flat one: activation must be declared separately.
        let bf16 = m.requirements.footprints.first { $0.quant == .bf16 }
        XCTAssertNotNil(bf16)
        XCTAssertGreaterThan(bf16?.peakActivationBytes ?? 0, 0)
    }

    /// The defaults ARE the contract for a distilled model: 4 steps on the packaged shift-3
    /// schedule with guidance internalized. A drifted default silently degrades every render.
    func testDistillationDefaults() {
        let cfg = QwenImageFlashConfiguration()
        XCTAssertEqual(cfg.defaultSteps, 4)
        XCTAssertEqual(cfg.defaultTrueCFGScale, 1.0)
        XCTAssertEqual(
            QwenImagePipeline.staticShiftedSigmas(steps: cfg.defaultSteps, shift: 3.0),
            [1.0, 0.9, 0.75, 0.5, 0.0])
    }

    // MARK: - MAT gate (offline)

    func testMATGate() {
        // Fresh machine: no explicit snapshot, no store root -> everything is missing.
        let fresh = QwenImageFlashConfiguration()
        // Satisfied: the explicit local snapshot short-circuits the store probe.
        let satisfied = QwenImageFlashConfiguration(snapshotPath: Self.snapshot)
        let report = MaterializationConformance.check(
            freshConfiguration: fresh,
            satisfiedConfiguration: FileManager.default.fileExists(atPath: Self.snapshot)
                ? satisfied : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesCoverTheWholePipeline() {
        let roles = Set(QwenImageFlashConfiguration().weightSources.map(\.role))
        XCTAssertEqual(roles, ["transformer", "vae", "text-encoder", "pipeline-config"])
        // One upstream repo backs them all — the Swift loader reads the diffusers layout
        // directly, so bf16 needs no separately hosted conversion.
        XCTAssertEqual(
            Set(QwenImageFlashConfiguration().weightSources.map(\.repo)),
            [QwenImageFlashConfiguration.repo])
    }

    // MARK: - live smoke

    func testLoadRunUnload() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["QIF_PKG"] == "1", "QIF_PKG=1")

        let package = QwenImageFlashPackage(configuration: .init(snapshotPath: Self.snapshot))
        let loadStart = Date()
        try await package.load()
        print(String(format: "load: %.1f s", -loadStart.timeIntervalSinceNow))

        let request = T2IRequest(
            prompt: "A red fox in a snowy pine forest at golden hour, photorealistic, "
                + "sharp focus, soft bokeh",
            width: 1024,
            height: 1024,
            seed: 42)
        let start = Date()
        let response = try await package.run(request)
        guard let t2i = response as? T2IResponse else {
            return XCTFail("wrong response type")
        }
        print(
            "package t2i: \(t2i.image.width ?? 0)x\(t2i.image.height ?? 0) in "
                + String(format: "%.1f s", -start.timeIntervalSinceNow))
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/qwen-image-flash-package-demo.png")
        try t2i.image.data.write(to: out)
        print("saved \(out.path)")

        await package.unload()
    }
}
