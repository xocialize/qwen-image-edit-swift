# Efficiency Adoption Brief — `qwen-image-edit-swift` (Qwen-Image-Edit-2511, `imageEdit`)

> **For a session-specific agent.** Adopt the engine 1.14 efficiency contract (engine 0.15.0). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (the four levers, "Gotchas &
> measurement", "Measurement findings", "Writing the brief") + references/memory-harness.md. Template:
> the LTX brief (`ltx-2-mlx-swift/EFFICIENCY-ADOPTION.md`) — this is the closest analog (multi-component,
> per-stage eviction is the headline). Audited 2026-06-30.

## Package at a glance
- **Multi-wrapper, one shared core.** Wrappers: `MLXQwenImageEdit` (`QwenImageEditPackage`, base) ·
  `MLXQwenImageEditTurbo` (Turbo, bf16/int4) · `MLXTeleStyle` (style-transfer). All three drive the
  shared core `QwenImageEdit` (`Pipeline.swift` / `QwenImageEditGenerator`). Capability `imageEdit`.
- **Three components, multi-GB:** `QwenVLPromptEncoder` (Qwen2.5-VL text/image encoder) + DiT
  (`QwenImageTransformer2DModel`) + VAE (`QwenImageVAE`). `load()` builds **all three up front** and
  holds them in the `Generator` for the package lifetime; `run()` → `generator.generate()` does
  encode → denoise → decode.
- **Footprints today (all FLAT):** base bf16 **60 GB** · TeleStyle bf16 **60 GB** · Turbo bf16 **57 GB** /
  int4 **22 GB**. No split, no `QuantConfigured`.
- **Why it's the next target:** most-consumed generation surface, and the first real **per-stage
  eviction** test since LTX — the encoder is idle through the denoise peak.

## Engine dependency status
- `Package.swift` pins `mlx-engine-swift` **`from: "0.3.0"`** (very old). **P0 = `swift package update`**
  → 0.15.0 (the pin admits it). The contract has grown a lot since 0.3.0 — after re-resolve, **build and
  fix any API drift** before the lever work (imageEdit/IEditRequest have existed since 1.2.0, so the core
  surface is stable, but verify).

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0; re-resolve to 0.15.0, fix any drift | **P0** |
| 1. Split footprint | ❌ | flat 60 GB (per wrapper) — no persistent/transient split, no QuantConfigured | **P1** |
| 2. Per-stage evict | ❌ | `load()` holds encoder+DiT+VAE resident the whole run; encoder used once then idle through denoise | **P2 (headline)** |
| 3. mmap/lazy | 🟡 verify | `loadDiTFromPT`/`loadVAE`/encoder loaders — check for eager full casts (DiT is the big one) | P3 |
| 4. BudgetAware | ➖ | Turbo has bf16/int4 but quant is config-chosen; defer | defer |

---

## P2 — Per-stage load → use → evict  (the headline; effort M–L)

`QwenImageEditPackage.load()` (and the Turbo/TeleStyle equivalents) build `QwenImageEditGenerator(encoder:transformer:vae:)`
with all three resident; `generate()` (core `Pipeline`/`Generator`) runs encode → denoise → decode. The
**Qwen2.5-VL encoder is used once** to encode the prompt + conditioning images, then **idle through the
entire DiT denoise + VAE decode** — the LTX-Gemma / Wan-T5 pattern.

- **Refactor the core `Generator`/`Pipeline`** to stage: load encoder → encode → **evict encoder
  (`nil` + `Memory.clearCache()`)** before the denoise loop; the VAE can load after denoise / evict after
  decode. Generalize the Wan `withTextEncoder` helper. Because the three wrappers share this core, **all
  three benefit from one refactor.**
- **Swift 6 isolation gotcha (from LTX):** if making the staged methods `async` (the encoder loader is
  async) "sends" the `@InferenceActor`-isolated, non-Sendable generator off-actor, fix with
  `isolated (any Actor)? = #isolation` on the async methods. Expect this here.
- **Tradeoff:** evicting the encoder means re-loading it per request (encode is cheap vs denoise). A
  keep-resident flag for big-RAM tiers is the natural refinement; evict-between-stages is the default.
- After P2, the denoise-peak residency drops (encoder no longer co-resident) → lower declared footprint.

## P1 — Declare the split  (effort S once P2 lands)

Conform the configs (`QwenImageEditConfiguration`, `QwenImageEditTurboConfiguration` [bf16/int4],
`TeleStyleConfiguration`) to **`QuantConfigured`** (add `var quant` if absent; Turbo's two quants make
this matter). Then declare each `QuantFootprint` as the split:
- **`residentBytes` (persistent weights)** is cheaply measurable now — sum the component weight bytes on
  disk for the *resident* set (after P2, the peak-phase resident = DiT + VAE, encoder evicted).
- **`peakActivationBytes` = peak − resident floor** — measure the activation high-water.

> **Measurement caveat (important).** This is a 60 GB-class model and its category app (`MLXEngineImage`)
> is **not cleanly stood up** (see the mlxengine-implementation skill — empty workspace). So the LTX-style
> "app headless autorun" measurement path is unavailable here. Use the package's **own smoke/CLI target via
> `xcodebuild`** with **weight prewarm** (cold-loading 60 GB off disk inside a live Metal buffer can trip
> the GPU watchdog — prewarm the files first, mirror LTX). If a full activation measurement isn't reliably
> obtainable, declare `residentBytes` from the measured weight floor (solid) and a **best-effort
> `peakActivationBytes`** (from a smoke run if it survives, else a conservative estimate) and **flag it for
> a clean app-autorun re-measure once the image app is stood up.** The split + P2 are still worth landing.

## Defer — P3 (mmap, verify only — DiT eager-cast check), P4 (BudgetAware: quant is config-chosen).

## Already good (don't regress)
- `ModelStorable` + model-store download path; cancellation (`Task.checkCancellation()` around generate);
  multi-image fusion; the three wrappers' surfaces (base/Turbo/TeleStyle).

## Definition of done
- [ ] `swift package update` → engine 0.15.0; build green (fix any drift since 0.3.0).
- [ ] Configs conform to `QuantConfigured` (esp. Turbo bf16/int4).
- [ ] Core `Generator`/`Pipeline` stages encoder load→encode→evict (before denoise) + defers/evicts VAE; `clearCache()` between stages; all three wrappers inherit it.
- [ ] Split declared per wrapper/quant (`residentBytes` weights floor + `peakActivationBytes`); activation measured-or-flagged per the caveat.
- [ ] Parity/smoke gates green; a smoke run (prewarmed) produces a valid edited image; record the split (+ flag if activation is best-effort).
- [ ] Registry: flip the qwen-image-edit row Eff ⬜→✅ (or 🔵 if activation is flagged-pending), Eng→0.15.0.

## Report back
Per wrapper: flat→split footprints, the denoise-peak drop from P2, effort (P0 drift fix + the Generator
refactor are the variables), the measurement path actually used (smoke vs flagged), any contract drift
since 0.3.0, and the commit SHAs. STAY IN SCOPE — only the four-lever adoption + brief + registry row;
do NOT touch testing apps or restructure anything; stop-and-report if a bigger change seems needed.

---

## Adoption outcome (executed 2026-06-30, engine 0.15.0)

**P0 — engine 0.15.0.** `swift package update mlx-engine-swift` moved the resolved engine 0.10.0 → **0.15.0**
(the `from: "0.3.0"` floor already admitted it; no manifest edit). **Zero API drift** — all three wrappers
(`MLXQwenImageEdit` / `MLXQwenImageEditTurbo` / `MLXTeleStyle`) built green against 0.15.0 unchanged. The
`imageEdit` / `IEditRequest` / `IEditResponse` / `PackageManifest` / `QuantFootprint(quant:residentBytes:)`
surface is stable from 0.3.0 → 0.15.0 (the new `peakActivationBytes` param defaults, so old call sites compile).

**P2 — per-stage encoder eviction (the headline).** Refactored the shared core `QwenImageEditGenerator`
(`Sources/QwenImageEdit/Pipeline.swift`): it no longer holds the Qwen2.5-VL encoder resident — it owns an
async `encoderProvider` closure (the wrapper's loader). `generate(...)` is now `async`: load encoder →
encode pos/neg → `eval` the embeddings → drop the encoder ref (`encoderRef = nil`) + `Memory.clearCache()`
→ then the DiT denoise loop + VAE decode. The DiT (with its bound LoRA swapper) + the small fp32 VAE stay
resident. **Swift 6 isolation:** the async `generate`/`generate(image:)`/`loadEncoder` take
`isolation: isolated (any Actor)? = #isolation`, so they inherit each wrapper's `@InferenceActor` and the
non-Sendable generator never crosses an actor hop (the canonical per-stage-eviction fix from LTX). A
back-compat `init(encoder:transformer:vae:)` keeps the encoder resident (`keepEncoderResident = true`) for
the parity tests, which can't reload it. All three wrappers were switched to the `encoderProvider` init →
all inherit the eviction from one refactor. **Parity preserved:** the encode/denoise/decode math is byte-
identical; only `eval()` (forces materialization, no numerical change) + the eviction were added. Test call
sites updated to `try await`.

**P1 — split footprint (MEASURED, M5 Max/128 GB, 1024²/8-step CFG4, seed 7, `QIE_MEMBENCH` in
MLXQwenImageEditTests; weights prewarmed via `cat`, no watchdog trip):**

| Wrapper / quant | OLD flat resident | Resident floor (declared) | Activation (declared) | Measured worst peak | Encoder now |
|---|---|---|---|---|---|
| base bf16 | 60 GB | **42 GB** (DiT 40.9 + fp32 VAE 0.5 = 41.4) | **21 GB** (peak−floor 17.9 +20%) | 59.2 GB | transient |
| TeleStyle bf16 | 60 GB | **42 GB** (same base weights) | **21 GB** | (= base) | transient |
| Turbo bf16 | 57 GB | **42 GB** (same base + Lightning rank factors) | **21 GB** | (= base) | transient |
| Turbo int4 | 22 GB | **12 GB** (int4 DiT ~11 + fp32 VAE 0.5) | **17 GB** (DERIVED + flagged) | ~29 GB (documented) | transient |

The headline: the **~16.6 GB encoder moved from resident into the transient bucket** — the persistent floor
drops from the flat 60 GB to **41.4 GB** measured. The engine now reserves ONE ~21 GB activation across
co-residents (`Σ residentBytes + max(peakActivationBytes)`) instead of baking the full 60 GB peak into each.
All four configs conform to `QuantConfigured` (Turbo's `quant` is computed: int4 when DiT is 4-bit, else bf16).

**Measurement path:** the package's own gated XCTest mem-bench (`QIE_MEMBENCH=1`), prewarmed — NOT the app
autorun (the image category app isn't stood up). The **bf16 split is freshly measured**; the **int4
activation is DERIVED** from the documented pre-split component measurements (resident 22 GB included the
co-resident encoder) and is **FLAGGED for a clean re-measure** with `quantizedDiTPath`+`quantizedEncoderPath`
once the image app supports app-autorun.

**P3 — mmap/lazy: verified, no change.** `loadDiTFromPT`/`loadVAE` rebuild a per-key dict with `v.asType(dtype)`
(lazy in MLX) and `eval(model)` once. The measured resident floor (41.4 GB) equals the on-disk DiT+VAE bytes
exactly (40.86 + 0.5), proving there is no full eager copy — the rebuild only rekeys mmap'd lazy arrays.

**P4 — BudgetAware: deferred.** Quant is config-chosen (bf16 vs the int4 pre-quantized tier), not a load-time
adaptive lever; no in-variant dtype/quality knob to drive from headroom.
