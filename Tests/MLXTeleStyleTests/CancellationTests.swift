// CancellationTests.swift — TeleStyle (content+style transfer on the QwenImageEdit core)
// through the engine's CAN gate (offline, no MLX kernels, no weights). CAN-1/2 drive the
// real run() pre-cancelled: the entry checkpoint (`try Task.checkCancellation()` as the
// FIRST act of run(), before notLoaded validation) fires before weights are touched.
// CAN-3 is the document of record for the checkpoint cadence — the checkpoints live in the
// SHARED core (QwenImageEditGenerator.generate, Sources/QwenImageEdit/Pipeline.swift):
// post-encode seam, per-denoise-step `try Task.checkCancellation()`, pre-decode seam; the
// throwing seam rethrows CancellationError unchanged (no laundering catch in this target).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXTeleStyle

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation (incl. the noImages guard) or weights are touched — offline-safe.
        let package = TeleStylePackage(configuration: TeleStyleConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // 21 GB declared peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: TeleStylePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: TeleStylePackage.manifest,
            posture: .cadence([
                // Per-denoise-step try Task.checkCancellation() in the shared core's denoise
                // loop (QwenImageEditGenerator.generate); post-encode + pre-decode seams
                // bracket it (single forwards — seams, not recurring units).
                .init(phase: .denoise, unit: .step),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
