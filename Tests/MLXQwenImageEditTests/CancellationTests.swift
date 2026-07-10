// CancellationTests.swift — QwenImageEdit (base 2511 tier) through the engine's CAN gate
// (offline, no MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled: the
// entry checkpoint (`try Task.checkCancellation()` as the FIRST act of run(), before
// notLoaded validation) fires before weights are touched, so a stub configuration suffices.
// CAN-3 is the document of record for the checkpoint cadence — all checkpoints live in the
// SHARED core (QwenImageEditGenerator.generate, Sources/QwenImageEdit/Pipeline.swift), a
// throwing seam that rethrows CancellationError unchanged:
//   - post-encode seam (VL prompt encoding done, encoder evicted, before denoise);
//   - denoise/step — `try Task.checkCancellation()` at the top of the denoise loop;
//   - pre-decode seam (before the monolithic VAE decode — ONE MLX eval, no chunk loop, so
//     no per-chunk decode cadence is claimed).
// The same core backs the Turbo + TeleStyle packages (and FireRed via checkpoint swap), so
// this one denoise-loop cadence covers all three; each package's own suite still runs the
// gate against its own PackageID.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXQwenImageEdit

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation (incl. the empty-images guard) or weights are touched — offline-safe.
        let package = QwenImageEditPackage(configuration: QwenImageEditConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // 21 GB declared peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: QwenImageEditPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: QwenImageEditPackage.manifest,
            posture: .cadence([
                // Per-denoise-step try Task.checkCancellation() in the shared core's denoise
                // loop (QwenImageEditGenerator.generate); post-encode + pre-decode seams
                // bracket it (single forwards — seams, not recurring units).
                .init(phase: .denoise, unit: .step),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
