# qwen-image-edit-swift

A Swift/MLX port of [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
plus its MLXEngine **`imageEdit`** packages — instruction-driven image editing (contract 1.3.0's
first `imageEdit` backer): identity-preserving edits, **multi-image fusion**, and **content+style
style transfer** (TeleStyleV2).

- **`QwenImageEdit`** — the standalone inference port: Qwen2.5-VL-7B prompt+image conditioning
  (via [qwen25vl-mlx-swift](https://github.com/xocialize/qwen25vl-mlx-swift)) → 20B 60-layer
  `zero_cond_t` double-stream DiT (Lens block family) → Wan 3D causal VAE. Reference = diffusers
  `QwenImageEditPlusPipeline` (the VL encoder runs **plain sequential 1D RoPE**, not the mRoPE grid
  — diffusers omits `mm_token_type_ids` so HF falls back; proven by true-SDPA-input capture).
  `generate(images:)` does N-image conditioning (per-image VAE cond latents + grids); image 0 is the
  content, later images are extra references ("Picture 1/2/…").
- **`MLXQwenImageEdit`** — the thin MLXEngine wrapper (`QwenImageEditPackage`, PackageID
  `qwen-image-edit`): the canonical `IEditRequest`/`IEditResponse` surface, multi-image, license
  declaration, requirements manifest, and PNG artifact encoding.
- **`MLXTeleStyle`** — [TeleStyleV2](https://github.com/Tele-AI/TeleStyleV2) content-preserving
  **style transfer** (`TeleStylePackage`, PackageID `telestyle-v2`): the `imageEdit` surface with a
  **`styleTransfer` mode** (image 0 = content, image 1 = style). Same `QwenImageEdit` core over a
  pre-fused snapshot (style + Lightning-4step DMD LoRAs merged at scale 1.0); 4-step DMD defaults.
  Weights: [`mlx-community/TeleStyleV2-Qwen-Image-Edit-2511-bf16`](https://huggingface.co/mlx-community/TeleStyleV2-Qwen-Image-Edit-2511-bf16).

## Parity

Validated against PyTorch fp32 goldens: DiT step-0 pos 0.99986 / neg 0.99944 bf16 (neg 0.999989
fp32-CPU) · VAE decode 73.7 dB · VAE encode 0.9999999 · VL encoder 0.974 bf16 / 0.9977 fp32-CPU.
In-app eye-verified: a lighthouse photo edited "dusk + stormy → day + clear", identity-preserving.

## Use

```swift
import MLXQwenImageEdit
import MLXToolKit

let package = QwenImageEditPackage(configuration: .init(
    snapshotPath: "<root>/Qwen-Image-Edit-2511"))   // transformer/ vae/ text_encoder/ processor/
try await package.load()
let response = try await package.run(IEditRequest(
    images: [inputImage],                            // schema is multi-image-first; core takes 1 for now
    prompt: "make it night with neon reflections",
    seed: 42)) as! IEditResponse                     // steps/true-CFG default to 20 / 4.0
// response.image: canonical Image (.png)
```

Behavior notes: the schema is **multi-image-first** (`images[]`, in prompt order — "Picture 1",
"Picture 2", …), but the core pipeline currently consumes a single conditioning image; multi-image
fusion (per-image VAE sizes + grids) is the tracked follow-up. `guidanceScale` is the true-CFG scale.

## Status / consuming this package

The **MLX-converted** weights are not yet on the Hub (the PyTorch source `Qwen/Qwen-Image-Edit-2511`
is; this wrapper needs MLX weights and reads a local snapshot — it does not download) — load from a
local `Qwen-Image-Edit-2511` snapshot. The package
depends on **`qwen25vl-mlx-swift`** (VL conditioning) and **`mlx-engine-swift`** (the `MLXToolKit`
contract) via tagged-URL net dependencies (`.package(url: "https://github.com/xocialize/qwen25vl-mlx-swift", from: "0.1.0")`
and `.package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0")`), so it builds
standalone. ~60 GB resident bf16 (20B DiT + VL-7B + fp32 VAE); 4-bit DiT+VL (~16 GB) is a tracked follow-up.

Apache-2.0 (weights) · MIT (port code).
