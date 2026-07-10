// CancellationTests.swift — QwenImageEditTurbo (Lightning/DMD low-step tier) through the
// engine's CAN gate (offline, no MLX kernels, no weights). CAN-1/2 drive the real run()
// pre-cancelled: the entry checkpoint (`try Task.checkCancellation()` as the FIRST act of
// run(), before notLoaded validation and the LoRA-effect swap) fires before weights are
// touched. CAN-3 is the document of record for the checkpoint cadence — the checkpoints
// live in the SHARED core (QwenImageEditGenerator.generate): post-encode seam, per-denoise-
// step `try Task.checkCancellation()`, pre-decode seam; the throwing seam rethrows
// CancellationError unchanged. CAN-2 laundering fix: LoRACache.ensure (LoRARegistry.swift)
// now rethrows CancellationError instead of wrapping it into LoRARegistryError.download.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXQwenImageEditTurbo

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config (base + Lightning 4-step DMD defaults); construction is cheap (C13)
        // and the entry checkpoint throws before validation or weights — offline-safe.
        let package = QwenImageEditTurboPackage(configuration: QwenImageEditTurboConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // 17–21 GB declared peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(
            CancellationConformance.longRunImplied(by: QwenImageEditTurboPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: QwenImageEditTurboPackage.manifest,
            posture: .cadence([
                // Per-denoise-step try Task.checkCancellation() in the shared core's denoise
                // loop (QwenImageEditGenerator.generate); post-encode + pre-decode seams
                // bracket it (single forwards — seams, not recurring units).
                .init(phase: .denoise, unit: .step),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
