// AutoencoderKLQwenImage — decoder-only Swift/MLX port (Wan-family 3D causal VAE).
//
// References: diffusers 0.37.1 autoencoder_kl_qwenimage.py (key/naming oracle; the
// checkpoint loads with diffusers names) and mflux's MLX implementation
// (models/qwen/model/qwen_vae — single-frame logic reference: time_conv is NOT
// applied for T=1 image decode; the checkpoint still carries its weights).
//
// Layout: internal tensors are channels-last (B, T, H, W, C) — MLX-native for
// Conv2d/Conv3d — with one transpose at decode() entry/exit to match the
// pipeline-facing PT convention (B, C, T, H, W).
//
// Norms are WanRMS: x / max(L2_norm_over_channels(x), eps) * sqrt(C) * gamma,
// eps 1e-12 — NOT mean-square RMSNorm. gamma is checkpointed as (C,1,1,1)/(C,1,1)
// and squeezed to (C) at load for the channels-last layout.

import Foundation
import MLX
import MLXExactConv
import MLXNN

/// Causal 3D conv: time is padded only at the front (2*pad), space symmetrically.
public final class QwenCausalConv3d: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d
    let padT: Int
    let padHW: Int

    public init(
        inChannels: Int, outChannels: Int, kernel: (Int, Int, Int), padding: (Int, Int)
    ) {
        self._conv.wrappedValue = Conv3d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: IntOrTriple([kernel.0, kernel.1, kernel.2]), padding: 0)
        self.padT = padding.0
        self.padHW = padding.1
        super.init()
    }

    /// x: (B, T, H, W, C)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if padT > 0 || padHW > 0 {
            x = padded(
                x,
                widths: [
                    IntOrPair([0, 0]), IntOrPair([2 * padT, 0]), IntOrPair([padHW, padHW]),
                    IntOrPair([padHW, padHW]), IntOrPair([0, 0]),
                ])
        }
        // Exact implicit GEMM: mlx-swift 0.32 splits small-depth conv3d into per-frame conv2d
        // that can take the lossy Winograd path (mlx#3785); see WinogradFreeConv2d.swift.
        return ExactConv.conv3d(x, weight: conv.weight, bias: conv.bias)
    }
}

/// WanRMS_norm: L2-over-channels normalization scaled by sqrt(C) * gamma.
public final class WanRMSNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float
    let eps: Float

    public init(channels: Int, eps: Float = 1e-12) {
        self._gamma.wrappedValue = MLXArray.ones([channels])
        self.scale = Float(channels).squareRoot()
        self.eps = eps
        super.init()
    }

    /// x: (..., C) — channels last.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let l2 = sqrt(sum(x * x, axis: -1, keepDims: true))
        let denom = maximum(l2, MLXArray(eps).asType(l2.dtype))
        return x / denom * scale * gamma
    }
}

public final class WanResBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: WanRMSNorm
    @ModuleInfo(key: "conv1") var conv1: QwenCausalConv3d
    @ModuleInfo(key: "norm2") var norm2: WanRMSNorm
    @ModuleInfo(key: "conv2") var conv2: QwenCausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: QwenCausalConv3d?

    public init(inChannels: Int, outChannels: Int) {
        self._norm1.wrappedValue = WanRMSNorm(channels: inChannels)
        self._conv1.wrappedValue = QwenCausalConv3d(
            inChannels: inChannels, outChannels: outChannels, kernel: (3, 3, 3), padding: (1, 1))
        self._norm2.wrappedValue = WanRMSNorm(channels: outChannels)
        self._conv2.wrappedValue = QwenCausalConv3d(
            inChannels: outChannels, outChannels: outChannels, kernel: (3, 3, 3), padding: (1, 1))
        self._convShortcut.wrappedValue = inChannels != outChannels
            ? QwenCausalConv3d(
                inChannels: inChannels, outChannels: outChannels, kernel: (1, 1, 1),
                padding: (0, 0))
            : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = convShortcut.map { $0(x) } ?? x
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        return h + residual
    }
}

/// Single-head 2D self-attention applied per frame (mid-block only).
///
/// Its (h·w)² score matrix grows with the square of the latent area: 1.07 GB fp32 at 1024², 3.0 GB
/// at 1328², 17.2 GB at 2048² (AB-L-0176). With `queryChunk` set (`QwenImageVAE.decodeBounded`
/// sets it on the decoder's mid block) the queries run in row chunks. Softmax is per query row, so
/// the formula is unchanged; only GEMM shapes differ (relL2 ≈ 7e-7). With nil (the default, and
/// every encode) it is the reference's single softmax(QKᵀ)V.
public final class WanAttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: WanRMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    @ModuleInfo(key: "proj") var proj: Conv2d
    let dim: Int
    /// Query rows per chunk, or nil for the single-matmul reference path.
    var queryChunk: Int?

    public init(dim: Int) {
        self.dim = dim
        self._norm.wrappedValue = WanRMSNorm(channels: dim)
        self._toQKV.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        self._proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }

    /// x: (B, T, H, W, C)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let identity = x
        var y = x.reshaped(b * t, h, w, c)
        y = norm(y)
        let qkv = toQKV(y).reshaped(b * t, h * w, 3, c)
        let q = qkv[0..., 0..., 0]
        let k = qkv[0..., 0..., 1]
        let v = qkv[0..., 0..., 2]
        let scale = 1.0 / sqrt(Float(c))
        let kT = k.transposed(0, 2, 1)
        let n = h * w
        var out: MLXArray
        if let queryChunk, n > queryChunk {
            var parts: [MLXArray] = []
            for r in stride(from: 0, to: n, by: queryChunk) {
                let scores = softmax(matmul(q[0..., r ..< min(r + queryChunk, n)], kT) * scale, axis: -1)
                let part = matmul(scores, v)
                eval(part)  // one chunk's scores at a time
                parts.append(part)
            }
            out = concatenated(parts, axis: 1).reshaped(b * t, h, w, c)
        } else {
            let scores = softmax(matmul(q, kT) * scale, axis: -1)
            out = matmul(scores, v).reshaped(b * t, h, w, c)
        }
        out = proj(out)
        return out.reshaped(b, t, h, w, c) + identity
    }
}

/// Upsampler: nearest-2x then Conv2d (checkpoint key `resample.1`). upsample3d
/// variants also checkpoint a `time_conv`, which is unused for single-frame decode.
/// The conv is a `WinogradFreeConv2d`: 384→192 and 192→96 are inside mlx's lossy Winograd
/// conv2d window. fp32 decode vs the CPU lane: 2.0e-3 → 7.6e-6 (256² golden), 2.2e-3 → 9.6e-5
/// (1024² photo; the rest is TF32 attention), for +12 ms on a ~1.1 s 1024² decode.
public final class WanUpsample: Module {
    @ModuleInfo(key: "resample") var resample: [Conv2d]
    @ModuleInfo(key: "time_conv") var timeConv: QwenCausalConv3d?

    public init(dim: Int, mode: String) {
        // upstream: resample = Sequential(Upsample(2x nearest), Conv2d) — index 1.
        // Our array has one Linear-position; sanitize maps `resample.1.` -> `resample.0.`.
        self._resample.wrappedValue = [
            WinogradFreeConv2d(
                inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, padding: 1)
        ]
        self._timeConv.wrappedValue = mode == "upsample3d"
            ? QwenCausalConv3d(
                inChannels: dim, outChannels: dim * 2, kernel: (3, 1, 1), padding: (1, 0))
            : nil
        super.init()
    }

    /// x: (B, T, H, W, C) — T=1 image path: nearest-2x spatial + conv, no time_conv.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var y = x.reshaped(b * t, h, w, c)
        y = repeated(repeated(y, count: 2, axis: 1), count: 2, axis: 2)
        y = resample[0](y)
        return y.reshaped(b, t, 2 * h, 2 * w, y.dim(-1))
    }
}

public final class WanMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [WanResBlock]
    @ModuleInfo(key: "attentions") var attentions: [WanAttentionBlock]

    public init(dim: Int) {
        self._resnets.wrappedValue = [
            WanResBlock(inChannels: dim, outChannels: dim),
            WanResBlock(inChannels: dim, outChannels: dim),
        ]
        self._attentions.wrappedValue = [WanAttentionBlock(dim: dim)]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = resnets[0](x)
        x = attentions[0](x)
        x = resnets[1](x)
        return x
    }
}

public final class WanUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [WanResBlock]
    @ModuleInfo(key: "upsamplers") var upsamplers: [WanUpsample]?

    public init(inChannels: Int, outChannels: Int, upsampleMode: String?) {
        var blocks: [WanResBlock] = []
        var dim = inChannels
        for _ in 0..<3 {  // num_res_blocks(2) + 1
            blocks.append(WanResBlock(inChannels: dim, outChannels: outChannels))
            dim = outChannels
        }
        self._resnets.wrappedValue = blocks
        self._upsamplers.wrappedValue = upsampleMode.map {
            [WanUpsample(dim: outChannels, mode: $0)]
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for r in resnets { x = r(x) }
        if let upsamplers { x = upsamplers[0](x) }
        return x
    }
}

public final class QwenImageVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: QwenCausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: WanMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [WanUpBlock]
    @ModuleInfo(key: "norm_out") var normOut: WanRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: QwenCausalConv3d

    public override init() {
        self._convIn.wrappedValue = QwenCausalConv3d(
            inChannels: 16, outChannels: 384, kernel: (3, 3, 3), padding: (1, 1))
        self._midBlock.wrappedValue = WanMidBlock(dim: 384)
        self._upBlocks.wrappedValue = [
            WanUpBlock(inChannels: 384, outChannels: 384, upsampleMode: "upsample3d"),
            WanUpBlock(inChannels: 192, outChannels: 384, upsampleMode: "upsample3d"),
            WanUpBlock(inChannels: 192, outChannels: 192, upsampleMode: "upsample2d"),
            WanUpBlock(inChannels: 96, outChannels: 96, upsampleMode: nil),
        ]
        self._normOut.wrappedValue = WanRMSNorm(channels: 96)
        self._convOut.wrappedValue = QwenCausalConv3d(
            inChannels: 96, outChannels: 3, kernel: (3, 3, 3), padding: (1, 1))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)
        x = midBlock(x)
        for block in upBlocks { x = block(x) }
        x = convOut(silu(normOut(x)))
        return x
    }
}

/// Decoder-only AutoencoderKLQwenImage. `decode` takes/returns the PT layout
/// (B, C, T, H, W); input latents must be ALREADY de-normalized (the pipeline
/// applies latents_mean/std before calling decode, matching diffusers).
public final class QwenImageVAE: Module {
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: QwenCausalConv3d
    @ModuleInfo(key: "quant_conv") var quantConv: QwenCausalConv3d
    @ModuleInfo(key: "decoder") var decoder: QwenImageVAEDecoder
    @ModuleInfo(key: "encoder") var encoder: QwenImageVAEEncoder

    /// Compute dtype of the loaded weights (set by `loadVAE`). decode/encode cast their
    /// input to this so a bf16 VAE runs the whole 1024² decode in bf16 (half the
    /// intermediates); defaults to fp32, the parity-locked regime.
    public var weightDtype: DType = .float32

    /// Route for the decoder's in-window upsampler convs (WinogradFreeConv2d.swift). Default
    /// `.conv3d`: exact, and ~free here (3 convs). `.winograd` is mlx's raw path (A/B only).
    public var convRoute: QwenVAEConvRoute {
        get { modules().lazy.compactMap { ($0 as? WinogradFreeConv2d)?.route }.first ?? .conv3d }
        set { for case let conv as WinogradFreeConv2d in modules() { conv.route = newValue } }
    }

    /// From vae/config.json (latents_mean / latents_std).
    public static let latentsMean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    public static let latentsStd: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ]

    public override init() {
        self._postQuantConv.wrappedValue = QwenCausalConv3d(
            inChannels: 16, outChannels: 16, kernel: (1, 1, 1), padding: (0, 0))
        self._quantConv.wrappedValue = QwenCausalConv3d(
            inChannels: 32, outChannels: 32, kernel: (1, 1, 1), padding: (0, 0))
        self._decoder.wrappedValue = QwenImageVAEDecoder()
        self._encoder.wrappedValue = QwenImageVAEEncoder()
        super.init()
        if let route = QwenVAEConvRoute.environmentOverride { convRoute = route }
    }

    /// latents: (B, 16, T, H, W) de-normalized -> image (B, 3, T, 8H, 8W).
    public func decode(_ latents: MLXArray) -> MLXArray {
        var x = latents.asType(weightDtype).transposed(0, 2, 3, 4, 1)  // -> (B, T, H, W, C)
        x = postQuantConv(x)
        x = decoder(x)
        return x.transposed(0, 4, 1, 2, 3)  // -> (B, C, T, H, W)
    }

    /// Latent token count (h·w) up to which `decodeBounded` is plain `decode`. The default 160²
    /// covers every 1024-bucket size and every ~1 MP edit output (≤ 18,240 tokens), so the gated
    /// paths never change. `Int.max` disables bounding (the v0.9.0 decode at every size).
    public var untiledDecodeMaxTokens = 160 * 160

    /// `decode` in bounded memory above `untiledDecodeMaxTokens` (AB-L-0176). Two levers, and it
    /// needs both. On its own each saves nothing, because the other one becomes the peak:
    /// - the mid block's global attention runs in 4096-row query chunks (≈ 1 GB of scores each at
    ///   2048²). Its (h·w)² fp32 scores are 17.2 GB at 2048² unchunked;
    /// - the up path, 27.7 GB at 2048² untiled, is tiled over the latent grid.
    ///
    /// Exact up to GEMM shape. The mid block runs on the whole latent. Everything after it is local:
    /// causal 3×3×3 convs at T = 1, per-pixel WanRMS norms, nearest-2× upsampling. Its receptive
    /// radius is ≈ 12.3 latent pixels: 6 convs at 1×, then 1 + 6 at each of 2×, 4× and 8×, then
    /// conv_out. Each tile carries a `halo` ≥ that radius of real neighbouring latent and keeps only
    /// its interior. The recipe is ming-image-swift's `MingVAE.decodeTiled`.
    public func decodeBounded(_ latents: MLXArray, tile: Int = 96, halo: Int = 16) -> MLXArray {
        let (h, w) = (latents.dim(3), latents.dim(4))
        guard h * w > untiledDecodeMaxTokens else { return decode(latents) }
        var x = latents.asType(weightDtype).transposed(0, 2, 3, 4, 1)  // -> (B, T, H, W, C)
        let attention = decoder.midBlock.attentions[0]
        attention.queryChunk = 4096
        x = decoder.midBlock(decoder.convIn(postQuantConv(x)))
        eval(x)
        attention.queryChunk = nil
        let s = 8  // latent → pixel scale of the up path
        var rows: [MLXArray] = []
        for y0 in stride(from: 0, to: h, by: tile) {
            var cols: [MLXArray] = []
            for x0 in stride(from: 0, to: w, by: tile) {
                let (ys, ye) = (max(0, y0 - halo), min(h, y0 + tile + halo))
                let (xs, xe) = (max(0, x0 - halo), min(w, x0 + tile + halo))
                var t = x[0..., 0..., ys ..< ye, xs ..< xe, 0...]
                for block in decoder.upBlocks {
                    t = block(t)
                    eval(t)  // one up block's intermediates at a time
                }
                t = decoder.convOut(silu(decoder.normOut(t)))
                let (ky, kx) = ((y0 - ys) * s, (x0 - xs) * s)
                let (ny, nx) = (min(tile, h - y0) * s, min(tile, w - x0) * s)
                let kept = t[0..., 0..., ky ..< (ky + ny), kx ..< (kx + nx), 0...]
                eval(kept)  // bound the transient to this tile
                cols.append(kept)
            }
            rows.append(cols.count == 1 ? cols[0] : concatenated(cols, axis: 3))
        }
        let full = rows.count == 1 ? rows[0] : concatenated(rows, axis: 2)
        return full.transposed(0, 4, 1, 2, 3)  // -> (B, C, T, H, W)
    }

    /// Pipeline-side de-normalization: packedLatents (B, 16, T, H, W) normalized.
    public static func deNormalize(_ latents: MLXArray) -> MLXArray {
        let mean = MLXArray(latentsMean).reshaped(1, 16, 1, 1, 1)
        let std = MLXArray(latentsStd).reshaped(1, 16, 1, 1, 1)
        return latents * std + mean
    }
}

// MARK: - Encoder (Wan-family, flat down_blocks list)

/// One entry of the FLAT `encoder.down_blocks` ModuleList: either a resnet or a
/// spatial downsample (asymmetric (0,1) pad + stride-2 conv; downsample3d entries
/// also checkpoint a time_conv, unused for single-frame encode).
public final class WanDownEntry: Module {
    @ModuleInfo(key: "norm1") var norm1: WanRMSNorm?
    @ModuleInfo(key: "conv1") var conv1: QwenCausalConv3d?
    @ModuleInfo(key: "norm2") var norm2: WanRMSNorm?
    @ModuleInfo(key: "conv2") var conv2: QwenCausalConv3d?
    @ModuleInfo(key: "conv_shortcut") var convShortcut: QwenCausalConv3d?
    @ModuleInfo(key: "resample") var resample: [Conv2d]?
    @ModuleInfo(key: "time_conv") var timeConv: QwenCausalConv3d?

    public init(resnetIn: Int, resnetOut: Int) {
        self._norm1.wrappedValue = WanRMSNorm(channels: resnetIn)
        self._conv1.wrappedValue = QwenCausalConv3d(
            inChannels: resnetIn, outChannels: resnetOut, kernel: (3, 3, 3), padding: (1, 1))
        self._norm2.wrappedValue = WanRMSNorm(channels: resnetOut)
        self._conv2.wrappedValue = QwenCausalConv3d(
            inChannels: resnetOut, outChannels: resnetOut, kernel: (3, 3, 3), padding: (1, 1))
        self._convShortcut.wrappedValue = resnetIn != resnetOut
            ? QwenCausalConv3d(
                inChannels: resnetIn, outChannels: resnetOut, kernel: (1, 1, 1), padding: (0, 0))
            : nil
        self._resample.wrappedValue = nil
        self._timeConv.wrappedValue = nil
        super.init()
    }

    public init(downsample dim: Int, temporal: Bool) {
        self._norm1.wrappedValue = nil
        self._conv1.wrappedValue = nil
        self._norm2.wrappedValue = nil
        self._conv2.wrappedValue = nil
        self._convShortcut.wrappedValue = nil
        self._resample.wrappedValue = [
            Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2, padding: 0)
        ]
        self._timeConv.wrappedValue = temporal
            ? QwenCausalConv3d(
                inChannels: dim, outChannels: dim, kernel: (3, 1, 1), padding: (0, 0))
            : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let resample {
            let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
            var y = x.reshaped(b * t, h, w, c)
            y = padded(
                y,
                widths: [
                    IntOrPair([0, 0]), IntOrPair([0, 1]), IntOrPair([0, 1]), IntOrPair([0, 0]),
                ])
            y = resample[0](y)
            return y.reshaped(b, t, y.dim(1), y.dim(2), y.dim(3))
        }
        let residual = convShortcut.map { $0(x) } ?? x
        var h = conv1!(silu(norm1!(x)))
        h = conv2!(silu(norm2!(h)))
        return h + residual
    }
}

public final class QwenImageVAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: QwenCausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [WanDownEntry]
    @ModuleInfo(key: "mid_block") var midBlock: WanMidBlock
    @ModuleInfo(key: "norm_out") var normOut: WanRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: QwenCausalConv3d

    public override init() {
        self._convIn.wrappedValue = QwenCausalConv3d(
            inChannels: 3, outChannels: 96, kernel: (3, 3, 3), padding: (1, 1))
        self._downBlocks.wrappedValue = [
            WanDownEntry(resnetIn: 96, resnetOut: 96),
            WanDownEntry(resnetIn: 96, resnetOut: 96),
            WanDownEntry(downsample: 96, temporal: false),
            WanDownEntry(resnetIn: 96, resnetOut: 192),
            WanDownEntry(resnetIn: 192, resnetOut: 192),
            WanDownEntry(downsample: 192, temporal: true),
            WanDownEntry(resnetIn: 192, resnetOut: 384),
            WanDownEntry(resnetIn: 384, resnetOut: 384),
            WanDownEntry(downsample: 384, temporal: true),
            WanDownEntry(resnetIn: 384, resnetOut: 384),
            WanDownEntry(resnetIn: 384, resnetOut: 384),
        ]
        self._midBlock.wrappedValue = WanMidBlock(dim: 384)
        self._normOut.wrappedValue = WanRMSNorm(channels: 384)
        self._convOut.wrappedValue = QwenCausalConv3d(
            inChannels: 384, outChannels: 32, kernel: (3, 3, 3), padding: (1, 1))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)
        for block in downBlocks { x = block(x) }
        x = midBlock(x)
        x = convOut(silu(normOut(x)))
        return x
    }
}

extension QwenImageVAE {
    /// image: (B, 3, T, H, W) in [-1, 1] -> NORMALIZED latents (B, 16, T, H/8, W/8).
    /// Mirrors diffusers `_encode_vae_image(..., sample_mode="argmax")`: the latent
    /// dist mode = mean = first 16 of quant_conv's 32 channels, then (x - mean)/std.
    public func encode(_ image: MLXArray) -> MLXArray {
        var x = image.transposed(0, 2, 3, 4, 1)  // -> (B, T, H, W, C)
        x = encoder(x)
        x = quantConv(x)
        x = x.transposed(0, 4, 1, 2, 3)  // -> (B, 32, T, h, w)
        let mean16 = x[0..., ..<16]
        let m = MLXArray(Self.latentsMean).reshaped(1, 16, 1, 1, 1)
        let s = MLXArray(Self.latentsStd).reshaped(1, 16, 1, 1, 1)
        return (mean16 - m) / s
    }
}
