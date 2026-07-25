// CAN gate for the Qwen-Image-Flash textToImage package (offline, no kernels, no weights).
// CAN-1/2 drive the real run() pre-cancelled: the entry checkpoint is the FIRST act of run(),
// before notLoaded validation, so a stub configuration suffices. CAN-3 documents the cadence,
// which lives in the shared core (QwenImageT2IGenerator.generate, Sources/QwenImageEdit/
// PipelineT2I.swift) and rethrows CancellationError unchanged:
//   - post-encode seam (text encoding done, encoder evicted, before denoise);
//   - denoise/step — `try Task.checkCancellation()` at the top of each of the 4 steps;
//   - pre-decode seam (before the monolithic VAE decode — ONE MLX eval, so no per-chunk
//     decode cadence is claimed).
//
// Four steps is a short run in wall-clock terms, but each step is a full 20B forward at
// 1024² — far from sub-second — so no `.subSecondRuns` exemption is claimed.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXQwenImageFlash

final class FlashCancellationTests: XCTestCase {

    func testCANGatePreCancelledRun() async {
        let package = QwenImageFlashPackage(configuration: QwenImageFlashConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: QwenImageFlashPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: QwenImageFlashPackage.manifest,
            posture: .cadence([
                .init(phase: .denoise, unit: .step)
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
