// FireRed-Image-Edit-1.1 as an alternate `imageEdit` PackageID on the QwenImageEdit
// core (FR1). FireRed-1.1 is architecturally STOCK Qwen-Image-Edit-Plus — same
// QwenImageTransformer2DModel (20B) / Qwen2.5-VL encoder / AutoencoderKLQwenImage,
// same diffusers component layout (`transformer/ vae/ text_encoder/ processor/`) —
// so there is no port and no conversion: `QwenImageEditPackage` runs the FireRed
// snapshot as-is via `snapshotPath`. This registration only carries the correct
// manifest (FireRed provenance + surface identity); requirements/footprints are the
// 2511-measured numbers, valid because the graph is identical.
//
// Claimed edge over 2511 (self-reported REDEdit-Bench, GEdit avg 7.943 vs 7.877):
// identity consistency, portrait/makeup, multi-element fusion, CJK text. PROMOTION
// GATE: in-app A/B vs 2511 on our own prompt corpus before this becomes a default.
//
// CAVEATS
//  - TeleStyle LoRAs are 2511-trained; FireRed is a divergent fine-tune from the T2I
//    base — do NOT stack MLXTeleStyle adapters on these weights (keep TeleStyle
//    pinned to 2511).
//  - FireRed-1.0 had a known "CFG offset too large" issue; validate the true-CFG
//    scale (`guidedNoise`) against their reference before trusting non-default
//    guidance values on 1.1.
//  - The 8-step distilled variant (FireRed-Image-Edit-1.0-Distilled) and the FireRed
//    Lightning LoRA belong under Turbo semantics — base-specific, so the LoRA must
//    never be fused onto 2511. Registered separately once the base A/B passes.

import Foundation
import MLXToolKit
import QwenImageEdit

/// The FireRed-1.1 checkpoint's engine identity. Register with a
/// `QwenImageEditConfiguration` whose `snapshotPath` points at the FireRed snapshot.
public enum FireRedImageEdit {
    public static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "FireRedTeam/FireRed-Image-Edit-1.1", revision: "main", tier: 1),
            // Same architecture + dtype as Qwen-Image-Edit-2511 ⇒ the 2511-measured
            // split footprint applies verbatim (see QwenImageEditPackage.manifest).
            requirements: QwenImageEditPackage.manifest.requirements,
            specialties: [],
            surfaces: [
                IEditContract.descriptor(
                    name: "firered-image-edit",
                    summary: "FireRed-Image-Edit-1.1 instruction editing (stock Qwen-"
                        + "Image-Edit-Plus graph, divergent fine-tune): strengths in "
                        + "identity consistency, portrait/makeup, multi-element fusion, "
                        + "CJK text. 1024²-area output, 20-step true CFG. Opt-in "
                        + "step-residual cache via metaData stepCacheMode "
                        + "(conservative|aggressive).",
                    modes: []
                )
            ]
        )
    }

    /// Registration reusing the QwenImageEdit package type under the FireRed manifest —
    /// the multi-package-per-capability route (PackageID = "firered-image-edit").
    public static var registration: PackageRegistration {
        PackageRegistration(manifest: manifest) { config in
            guard let typed = config as? QwenImageEditConfiguration else {
                throw PackageError.configurationMismatch(
                    expected: String(describing: QwenImageEditConfiguration.self),
                    got: String(describing: type(of: config)))
            }
            return QwenImageEditPackage(configuration: typed)
        }
    }
}
