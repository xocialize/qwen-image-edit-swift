// MAT gate for the base 2511 edit package (offline).
//
// Retrofitted alongside the Flash tier: this package used to require a hand-staged local
// snapshot, so it worked only on a machine that already had the 54 GB sitting on disk. It now
// declares WeightSourcing against UPSTREAM `Qwen/Qwen-Image-Edit-2511` directly — no mirror was
// needed, because upstream ships `processor/tokenizer.json` (the file whose absence is exactly
// what forced an mlx-community mirror for Flash).
//
// Run: swift test --filter EditMaterializationTests

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXQwenImageEdit

final class EditMaterializationTests: XCTestCase {

    /// A hand-staged snapshot, if this machine happens to have one.
    static let localSnapshot = "/Volumes/DEV_ARCHIVE/models/Qwen/Qwen-Image-Edit-2511"

    func testMATGate() {
        // Fresh machine: default configuration (no explicit path), no store root ⇒ all missing.
        let fresh = QwenImageEditConfiguration()
        let satisfied = QwenImageEditConfiguration(snapshotPath: Self.localSnapshot)
        let report = MaterializationConformance.check(
            freshConfiguration: fresh,
            satisfiedConfiguration: FileManager.default.fileExists(atPath: Self.localSnapshot)
                ? satisfied : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesCoverTheWholePipeline() {
        let cfg = QwenImageEditConfiguration()
        XCTAssertEqual(
            Set(cfg.weightSources.map(\.role)),
            ["transformer", "vae", "text-encoder", "pipeline-config"])
        XCTAssertEqual(Set(cfg.weightSources.map(\.repo)), [QwenImageEditConfiguration.repo])
        // The VL encoder needs processor/ (tokenizer.json + preprocessor config) alongside the
        // weights — a text_encoder-only glob materializes a snapshot that cannot load.
        let globs = cfg.weightSources.flatMap { $0.matching ?? [] }
        XCTAssertTrue(globs.contains("processor/*"))
        XCTAssertTrue(globs.contains("transformer/*"))
        XCTAssertTrue(globs.contains("vae/*"))
    }

    /// The default configuration must route through the store; an explicit path pins a local
    /// snapshot and reports nothing missing (the escape hatch FireRed and the Turbo tier use).
    func testExplicitPathShortCircuitsMaterialization() {
        XCTAssertTrue(QwenImageEditConfiguration().snapshotPath.isEmpty)
        XCTAssertFalse(QwenImageEditConfiguration().missingWeightSources(storeRoot: nil).isEmpty)

        try? FileManager.default.createDirectory(
            at: URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("qie-mat-probe/transformer"),
            withIntermediateDirectories: true)
        let pinned = QwenImageEditConfiguration(
            snapshotPath: URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("qie-mat-probe").path)
        XCTAssertTrue(pinned.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(
            pinned.resolvedSnapshotDirectory(storeRoot: nil)?.lastPathComponent, "qie-mat-probe")
    }
}
