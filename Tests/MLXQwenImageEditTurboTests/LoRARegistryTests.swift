// Guards the bundled effect registry (Resources/qie-lora-registry.json): it must ship in the
// module bundle, decode, and be well-formed (unique ids, non-empty repos/files/triggers).
// Pure resource decode — no model load, no GPU — so it always runs.
//
// Run: swift test --filter LoRARegistryTests

import XCTest

@testable import MLXQwenImageEditTurbo

final class LoRARegistryTests: XCTestCase {
    func testBundledRegistryWellFormed() throws {
        let registry = try LoRARegistry.bundled()
        XCTAssertEqual(registry.schemaVersion, 1)
        XCTAssertFalse(registry.adapters.isEmpty)

        var ids = Set<String>()
        for a in registry.adapters {
            XCTAssertTrue(ids.insert(a.id).inserted, "duplicate adapter id '\(a.id)'")
            XCTAssertFalse(a.id.isEmpty, "empty id")
            XCTAssertFalse(a.displayName.isEmpty, "\(a.id): empty displayName")
            XCTAssertFalse(a.repo.isEmpty, "\(a.id): empty repo")
            XCTAssertFalse(a.weightFile.isEmpty, "\(a.id): empty weightFile")
            XCTAssertFalse(
                a.trigger.trimmingCharacters(in: .whitespaces).isEmpty,
                "\(a.id): empty trigger (every curated adapter needs a trigger prompt)")
            XCTAssertGreaterThan(a.defaultStrength, 0, "\(a.id): non-positive strength")
        }

        // Lookup works for a known id and fails cleanly for an unknown one.
        XCTAssertNotNil(registry.entry(id: "pixar-inspired-3d"))
        XCTAssertNil(registry.entry(id: "nope"))
    }
}
