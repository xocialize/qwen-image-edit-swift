// The multi-LoRA request contract for the T2I tier (offline; no weights, no kernels).
//
// The gallery UI hands the package up to four adapter slots per request, and a slot can point
// at ANY HuggingFace repo (the user can paste a custom one), so the request carries full
// coordinates rather than a registry id. These tests pin the wire format both directions and
// the malformed-slot policy — a bad slot must not cost the user a render.
//
// The live swap is gated separately (QIF_LORA_LIVE) since it loads the 20B DiT.

import Foundation
import MLXToolKit
import QwenImageEdit
import XCTest

@testable import MLXQwenImageFlash

final class FlashLoRAStackTests: XCTestCase {

    func testEncodeDecodeRoundTrip() {
        let stack = [
            LoRAStackItem(
                id: "day-of-the-tentacle", repo: "xocialize/qwen-image-flash-loras",
                file: "loras/day-of-the-tentacle.safetensors", strength: 1.0),
            LoRAStackItem(
                id: "panoramic-wide-fisheye", repo: "xocialize/qwen-image-flash-loras",
                file: "loras/panoramic-wide-fisheye.safetensors", strength: 0.75),
        ]
        let decoded = LoRAStackItem.decode(LoRAStackItem.encode(stack))
        XCTAssertEqual(decoded, stack)
    }

    /// A slot missing coordinates is skipped, not fatal — the rest of the stack still renders.
    func testMalformedSlotsAreSkippedNotFatal() {
        let value = MetaValue.array([
            .object(["repo": .string("org/good"), "file": .string("a.safetensors")]),
            .object(["repo": .string("org/no-file")]),
            .object(["file": .string("orphan.safetensors")]),
            .string("not an object"),
            .object(["repo": .string(""), "file": .string("empty-repo.safetensors")]),
        ])
        let decoded = LoRAStackItem.decode(value)
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded.first?.repo, "org/good")
        // Strength defaults to 1.0 when the slot omits it.
        XCTAssertEqual(decoded.first?.strength, 1.0)
        // Absent id falls back to a stable repo/file identity so the cache still keys correctly.
        XCTAssertEqual(decoded.first?.id, "org/good/a.safetensors")
    }

    func testAbsentOrEmptyStackDecodesEmpty() {
        XCTAssertTrue(LoRAStackItem.decode(nil).isEmpty)
        XCTAssertTrue(LoRAStackItem.decode(.array([])).isEmpty)
        XCTAssertTrue(LoRAStackItem.decode(.string("loraId")).isEmpty)
    }

    /// Strength accepts int or double on the wire (a UI slider may hand over either).
    func testStrengthAcceptsIntAndDouble() {
        let value = MetaValue.array([
            .object([
                "repo": .string("org/a"), "file": .string("a.safetensors"), "strength": .int(1),
            ]),
            .object([
                "repo": .string("org/b"), "file": .string("b.safetensors"),
                "strength": .double(0.6),
            ]),
        ])
        let decoded = LoRAStackItem.decode(value)
        XCTAssertEqual(decoded.map(\.strength), [1.0, 0.6])
    }

    /// The cache keys on `id`, so two slots from different repos never collide, and the same
    /// adapter re-selected across runs reuses one file.
    func testCacheEntryMapping() {
        let item = LoRAStackItem(
            id: "martha", repo: "xocialize/qwen-image-flash-loras",
            file: "loras/martha.safetensors", strength: 0.8)
        let entry = item.entry
        XCTAssertEqual(entry.id, "martha")
        XCTAssertEqual(entry.repo, "xocialize/qwen-image-flash-loras")
        XCTAssertEqual(entry.weightFile, "loras/martha.safetensors")
        let cache = LoRACache(directory: URL(fileURLWithPath: "/tmp/qif-lora-cache"))
        XCTAssertEqual(cache.localURL(for: entry).lastPathComponent, "martha.safetensors")
    }
}
