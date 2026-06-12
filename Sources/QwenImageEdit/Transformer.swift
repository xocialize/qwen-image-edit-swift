// Qwen-Image-Edit-2511 denoising transformer (DiT) — Swift/MLX port.
//
// Isomorphic to diffusers 0.37.1 transformer_qwenimage.py (QwenImageTransformer2DModel
// with zero_cond_t=true), adapted from the parity-locked lens-mlx-swift DiT family.
// Deltas vs Lens: LayerNorm(affine:false) block norms (Lens: RMSNorm), GELU-approx
// FeedForward (Lens: SwiGLU), separate per-stream QKV projections (Lens: fused),
// joint attention order [text, image] (Lens: [image, text]), QK-norm eps 1e-6
// (Lens: 1e-5), multi-image RoPE with per-image frame-index offsets, and the
// zero_cond_t doubled-temb / per-token indexed modulation.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum QwenImageEditError: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .loading(let m): return "QwenImageEdit loading error: \(m)"
        case .invalidInput(let m): return "QwenImageEdit input error: \(m)"
        }
    }
}

// MARK: - Embeddings & RoPE

/// Sinusoidal timestep embeddings (DDPM-style). Mirrors diffusers get_timestep_embedding.
func getTimestepEmbedding(
    timesteps: MLXArray,
    embeddingDim: Int,
    flipSinToCos: Bool = false,
    downscaleFreqShift: Float = 1.0,
    scale: Float = 1.0,
    maxPeriod: Int = 10000
) -> MLXArray {
    precondition(timesteps.ndim == 1, "Timesteps should be 1-D")
    let halfDim = embeddingDim / 2
    var exponent = -log(Float(maxPeriod)) * MLXArray(0..<halfDim).asType(.float32)
    exponent = exponent / (Float(halfDim) - downscaleFreqShift)
    var emb = exp(exponent)
    emb = timesteps[0..., .newAxis].asType(.float32) * emb[.newAxis, 0...]
    emb = scale * emb
    var out = concatenated([sin(emb), cos(emb)], axis: -1)
    if flipSinToCos {
        out = concatenated([out[0..., halfDim...], out[0..., ..<halfDim]], axis: -1)
    }
    return out
}

/// Complex RoPE (use_real=False in the reference) as a real interleaved-pair rotation.
/// x: [B, S, H, D]; cos/sin: [S, D/2] (real/imag angle tables of `freqs_cis`).
func applyRotaryEmbQwen(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = x.reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
    let xR = pairs[.ellipsis, 0]
    let xI = pairs[.ellipsis, 1]
    let c = cosT[.newAxis, 0..., .newAxis, 0...]
    let s = sinT[.newAxis, 0..., .newAxis, 0...]
    let outR = xR * c - xI * s
    let outI = xR * s + xI * c
    return stacked([outR, outI], axis: -1).reshaped(shape).asType(x.dtype)
}

public final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
        super.init()
    }

    public func callAsFunction(_ sample: MLXArray) -> MLXArray {
        linear2(silu(linear1(sample)))
    }
}

/// time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
/// followed by TimestepEmbedding(256 -> innerDim). use_additional_t_cond is false for 2511.
public final class QwenTimestepProjEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedding

    public init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue = TimestepEmbedding(
            inChannels: 256, timeEmbedDim: embeddingDim)
        super.init()
    }

    public func callAsFunction(_ timestep: MLXArray, _ hiddenStates: MLXArray) -> MLXArray {
        let proj = getTimestepEmbedding(
            timesteps: timestep, embeddingDim: 256, flipSinToCos: true,
            downscaleFreqShift: 0, scale: 1000)
        return timestepEmbedder(proj.asType(hiddenStates.dtype))
    }
}

/// Frame/H/W axial RoPE (QwenEmbedRope, scale_rope=true). Plain class — the tables
/// are computed, not checkpointed. Supports a LIST of image shapes: the target grid
/// plus one entry per conditioning image; entry i uses frame-frequency index i,
/// which is how multiple images are positionally separated on the frame axis.
public final class QwenEmbedRope {
    let theta: Int
    let axesDim: [Int]
    let scaleRope: Bool
    let posFreqs: MLXArray  // [4096, sum(axesDim)/2] angles
    let negFreqs: MLXArray

    public init(theta: Int, axesDim: [Int], scaleRope: Bool = true) {
        self.theta = theta
        self.axesDim = axesDim
        self.scaleRope = scaleRope
        let posIndex = MLXArray(0..<4096)
        let negIndex = MLXArray(0..<4096) - 4096  // == arange(4096)[::-1] * -1 - 1
        self.posFreqs = concatenated(
            axesDim.map { Self.ropeParams(index: posIndex, dim: $0, theta: theta) }, axis: 1)
        self.negFreqs = concatenated(
            axesDim.map { Self.ropeParams(index: negIndex, dim: $0, theta: theta) }, axis: 1)
    }

    static func ropeParams(index: MLXArray, dim: Int, theta: Int = 10000) -> MLXArray {
        precondition(dim % 2 == 0)
        let invFreq = 1.0 / pow(
            MLXArray(Float(theta)),
            MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim))
        return outer(index.asType(.float32), invFreq)
    }

    /// Returns ((imgCos, imgSin), (txtCos, txtSin)) for the concatenated image
    /// sequence (target tokens then conditioning tokens, in list order).
    public func callAsFunction(
        videoFHW: [(frame: Int, height: Int, width: Int)], txtSeqLen: Int
    ) -> ((MLXArray, MLXArray), (MLXArray, MLXArray)) {
        var vidFreqs: [MLXArray] = []
        var maxVidIndex = 0
        for (idx, fhw) in videoFHW.enumerated() {
            vidFreqs.append(
                computeVideoFreqs(frame: fhw.frame, height: fhw.height, width: fhw.width, idx: idx))
            let extent = scaleRope
                ? max(fhw.height / 2, fhw.width / 2) : max(fhw.height, fhw.width)
            maxVidIndex = max(extent, maxVidIndex)
        }
        let vid = concatenated(vidFreqs, axis: 0)
        let txt = posFreqs[maxVidIndex ..< (maxVidIndex + txtSeqLen), 0...]
        return ((cos(vid), sin(vid)), (cos(txt), sin(txt)))
    }

    func computeVideoFreqs(frame: Int, height: Int, width: Int, idx: Int) -> MLXArray {
        let seqLens = frame * height * width
        let splits = axesDim.map { $0 / 2 }
        var bounds: [Int] = [0]
        for s in splits { bounds.append(bounds.last! + s) }
        let fp = (0..<splits.count).map { posFreqs[0..., bounds[$0] ..< bounds[$0 + 1]] }
        let fn = (0..<splits.count).map { negFreqs[0..., bounds[$0] ..< bounds[$0 + 1]] }

        let freqsFrame = broadcast(
            fp[0][idx ..< (idx + frame)].reshaped(frame, 1, 1, -1),
            to: [frame, height, width, splits[0]])

        let freqsHeight: MLXArray
        let freqsWidth: MLXArray
        if scaleRope {
            freqsHeight = broadcast(
                concatenated([fn[1][(4096 - (height - height / 2))...], fp[1][..<(height / 2)]], axis: 0)
                    .reshaped(1, height, 1, -1),
                to: [frame, height, width, splits[1]])
            freqsWidth = broadcast(
                concatenated([fn[2][(4096 - (width - width / 2))...], fp[2][..<(width / 2)]], axis: 0)
                    .reshaped(1, 1, width, -1),
                to: [frame, height, width, splits[2]])
        } else {
            freqsHeight = broadcast(
                fp[1][..<height].reshaped(1, height, 1, -1),
                to: [frame, height, width, splits[1]])
            freqsWidth = broadcast(
                fp[2][..<width].reshaped(1, 1, width, -1),
                to: [frame, height, width, splits[2]])
        }
        return concatenated([freqsFrame, freqsHeight, freqsWidth], axis: -1)
            .reshaped(seqLens, -1)
    }
}

// MARK: - Feed-forward (GELU-approximate)

/// diffusers FeedForward(activation_fn="gelu-approximate"): net.0.proj -> GELU(tanh)
/// -> net.2. Keys sanitized to proj_in / proj_out at load.
public final class QwenFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._projIn.wrappedValue = Linear(dim, hiddenDim, bias: true)
        self._projOut.wrappedValue = Linear(hiddenDim, dim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(geluApproximate(projIn(x)))
    }
}

// MARK: - Attention (joint text + image)

/// QwenDoubleStreamAttnProcessor2_0: separate QKV per stream, RMSNorm QK, complex
/// RoPE, joint [text, image] SDPA, split, per-stream output projections.
public final class QwenJointAttention: Module {
    let heads: Int
    let dimHead: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear

    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

    // upstream: to_out = ModuleList([Linear, Dropout]); index 0 is the Linear.
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    public init(queryDim: Int, heads: Int, dimHead: Int, eps: Float = 1e-6) {
        self.heads = heads
        self.dimHead = dimHead
        let innerDim = heads * dimHead

        self._toQ.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._addQProj.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._addKProj.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._addVProj.wrappedValue = Linear(queryDim, innerDim, bias: true)

        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)

        self._toOut.wrappedValue = [Linear(innerDim, queryDim, bias: true)]
        self._toAddOut.wrappedValue = Linear(innerDim, queryDim, bias: true)
        super.init()
    }

    /// Returns (imgAttnOutput, txtAttnOutput).
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        imageRotaryEmb: ((MLXArray, MLXArray), (MLXArray, MLXArray)),
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let bsz = hiddenStates.dim(0)
        let seqImg = hiddenStates.dim(1)
        let seqTxt = encoderHiddenStates.dim(1)
        let (H, Dh) = (heads, dimHead)

        var imgQ = toQ(hiddenStates).reshaped(bsz, seqImg, H, Dh)
        var imgK = toK(hiddenStates).reshaped(bsz, seqImg, H, Dh)
        let imgV = toV(hiddenStates).reshaped(bsz, seqImg, H, Dh)
        var txtQ = addQProj(encoderHiddenStates).reshaped(bsz, seqTxt, H, Dh)
        var txtK = addKProj(encoderHiddenStates).reshaped(bsz, seqTxt, H, Dh)
        let txtV = addVProj(encoderHiddenStates).reshaped(bsz, seqTxt, H, Dh)

        imgQ = normQ(imgQ)
        imgK = normK(imgK)
        txtQ = normAddedQ(txtQ)
        txtK = normAddedK(txtK)

        let ((imgCos, imgSin), (txtCos, txtSin)) = imageRotaryEmb
        imgQ = applyRotaryEmbQwen(imgQ, cos: imgCos[..<seqImg], sin: imgSin[..<seqImg])
        imgK = applyRotaryEmbQwen(imgK, cos: imgCos[..<seqImg], sin: imgSin[..<seqImg])
        txtQ = applyRotaryEmbQwen(txtQ, cos: txtCos[..<seqTxt], sin: txtSin[..<seqTxt])
        txtK = applyRotaryEmbQwen(txtK, cos: txtCos[..<seqTxt], sin: txtSin[..<seqTxt])

        // Joint order is [text, image] (reference L544), then [B, H, S, D] for SDPA.
        let q = concatenated([txtQ, imgQ], axis: 1).transposed(0, 2, 1, 3)
        let k = concatenated([txtK, imgK], axis: 1).transposed(0, 2, 1, 3)
        let v = concatenated([txtV, imgV], axis: 1).transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(Dh))
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            attentionMask.map { .array($0) } ?? .none
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        out = out.transposed(0, 2, 1, 3).reshaped(bsz, seqTxt + seqImg, -1)

        let txtOut = toAddOut(out[0..., ..<seqTxt, 0...])
        let imgOut = toOut[0](out[0..., seqTxt..., 0...])
        return (imgOut, txtOut)
    }
}

// MARK: - Transformer block (zero_cond_t)

public final class QwenTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: QwenJointAttention

    @ModuleInfo(key: "img_mod") var imgMod: Linear  // upstream Sequential(SiLU, Linear) -> .1
    @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
    @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
    @ModuleInfo(key: "img_mlp") var imgMLP: QwenFeedForward

    @ModuleInfo(key: "txt_mod") var txtMod: Linear
    @ModuleInfo(key: "txt_norm1") var txtNorm1: LayerNorm
    @ModuleInfo(key: "txt_norm2") var txtNorm2: LayerNorm
    @ModuleInfo(key: "txt_mlp") var txtMLP: QwenFeedForward

    public init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, eps: Float = 1e-6) {
        self._attn.wrappedValue = QwenJointAttention(
            queryDim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim, eps: eps)

        self._imgMod.wrappedValue = Linear(dim, 6 * dim, bias: true)
        self._imgNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._imgNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._imgMLP.wrappedValue = QwenFeedForward(dim: dim, hiddenDim: 4 * dim)

        self._txtMod.wrappedValue = Linear(dim, 6 * dim, bias: true)
        self._txtNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._txtNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._txtMLP.wrappedValue = QwenFeedForward(dim: dim, hiddenDim: 4 * dim)
        super.init()
    }

    /// zero_cond_t modulation (reference `_modulate`, L628-662): modParams has batch
    /// 2B = [params(t), params(t=0)]; with an index (B, L), token l takes params(t)
    /// where index==0 (target latents) and params(t=0) where index==1 (conditioning).
    static func modulate(
        _ x: MLXArray, _ modParams: MLXArray, index: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let parts = split(modParams, parts: 3, axis: -1)
        let (shift, scale, gate) = (parts[0], parts[1], parts[2])
        guard let index else {
            return (
                x * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...],
                gate[0..., .newAxis, 0...]
            )
        }
        let b = shift.dim(0) / 2
        let idx = index[.ellipsis, .newAxis]  // (B, L, 1)
        func sel(_ p: MLXArray) -> MLXArray {
            MLX.which(idx .== 0, p[..<b][0..., .newAxis, 0...], p[b...][0..., .newAxis, 0...])
        }
        return (x * (1 + sel(scale)) + sel(shift), sel(gate))
    }

    /// temb: (2B, dim) = [temb(t), temb(0)]. txt stream uses the real-t half only
    /// (reference L677-679); the img stream selects per token via modulateIndex.
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        temb: MLXArray,
        imageRotaryEmb: ((MLXArray, MLXArray), (MLXArray, MLXArray)),
        attentionMask: MLXArray?,
        modulateIndex: MLXArray?
    ) -> (MLXArray, MLXArray) {
        var hiddenStates = hiddenStates
        var encoderHiddenStates = encoderHiddenStates
        let b = temb.dim(0) / 2

        let imgMods = split(imgMod(silu(temb)), parts: 2, axis: -1)
        let txtMods = split(txtMod(silu(temb[..<b])), parts: 2, axis: -1)

        let (imgModulated, imgGate1) = Self.modulate(
            imgNorm1(hiddenStates), imgMods[0], index: modulateIndex)
        let (txtModulated, txtGate1) = Self.modulate(
            txtNorm1(encoderHiddenStates), txtMods[0], index: nil)

        let (imgAttn, txtAttn) = attn(
            hiddenStates: imgModulated, encoderHiddenStates: txtModulated,
            imageRotaryEmb: imageRotaryEmb, attentionMask: attentionMask)

        hiddenStates = hiddenStates + imgGate1 * imgAttn
        encoderHiddenStates = encoderHiddenStates + txtGate1 * txtAttn

        let (imgModulated2, imgGate2) = Self.modulate(
            imgNorm2(hiddenStates), imgMods[1], index: modulateIndex)
        hiddenStates = hiddenStates + imgGate2 * imgMLP(imgModulated2)

        let (txtModulated2, txtGate2) = Self.modulate(
            txtNorm2(encoderHiddenStates), txtMods[1], index: nil)
        encoderHiddenStates = encoderHiddenStates + txtGate2 * txtMLP(txtModulated2)

        return (encoderHiddenStates, hiddenStates)
    }
}

// MARK: - Top-level model

/// norm_out: SiLU -> Linear(dim, 2*dim) -> affine-less LayerNorm modulation.
public final class AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    public init(dim: Int, condDim: Int, eps: Float = 1e-6) {
        self._linear.wrappedValue = Linear(condDim, 2 * dim, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let emb = linear(silu(conditioning))
        let parts = split(emb, parts: 2, axis: -1)
        let (scale, shift) = (parts[0], parts[1])
        return norm(x) * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...]
    }
}

/// QwenImageTransformer2DModel (zero_cond_t=true, guidance_embeds=false).
public final class QwenImageTransformer2DModel: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let innerDim: Int
    public let patchSize: Int

    public let posEmbed: QwenEmbedRope  // plain class — no parameters

    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: QwenTimestepProjEmbeddings
    @ModuleInfo(key: "txt_norm") var txtNorm: RMSNorm
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "img_in") var imgIn: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [QwenTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(
        patchSize: Int = 2,
        inChannels: Int = 64,
        outChannels: Int? = 16,
        numLayers: Int = 60,
        attentionHeadDim: Int = 128,
        numAttentionHeads: Int = 24,
        jointAttentionDim: Int = 3584,
        axesDimsRope: [Int] = [16, 56, 56]
    ) {
        let inner = numAttentionHeads * attentionHeadDim
        self.inChannels = inChannels
        self.outChannels = outChannels ?? inChannels
        self.innerDim = inner
        self.patchSize = patchSize

        self.posEmbed = QwenEmbedRope(theta: 10000, axesDim: axesDimsRope, scaleRope: true)
        self._timeTextEmbed.wrappedValue = QwenTimestepProjEmbeddings(embeddingDim: inner)
        self._txtNorm.wrappedValue = RMSNorm(dimensions: jointAttentionDim, eps: 1e-6)
        self._txtIn.wrappedValue = Linear(jointAttentionDim, inner)
        self._imgIn.wrappedValue = Linear(inChannels, inner)
        self._transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            QwenTransformerBlock(
                dim: inner, numAttentionHeads: numAttentionHeads,
                attentionHeadDim: attentionHeadDim)
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(dim: inner, condDim: inner)
        self._projOut.wrappedValue = Linear(
            inner, patchSize * patchSize * (outChannels ?? inChannels), bias: true)
        super.init()
    }

    /// - Parameters:
    ///   - hiddenStates: [B, L_img, inChannels] packed latents — target tokens first,
    ///     then the conditioning-image tokens in imgShapes order.
    ///   - encoderHiddenStates: [B, S, jointAttentionDim] VL features.
    ///   - encoderHiddenStatesMask: [B, S], 1 = real token; nil when all-ones.
    ///   - timestep: [B] sigma values in 0…1 (= reference timestep/1000). The
    ///     zero_cond_t doubling ([t, 0]) happens inside.
    ///   - imgShapes: [(frame, hPatches, wPatches)] — target grid first, then one
    ///     entry per conditioning image.
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderHiddenStatesMask: MLXArray?,
        timestep: MLXArray,
        imgShapes: [(Int, Int, Int)]
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let imgLen = hiddenStates.dim(1)
        let txtLen = encoderHiddenStates.dim(1)

        var hidden = imgIn(hiddenStates)
        var encoder = txtIn(txtNorm(encoderHiddenStates))

        // zero_cond_t: [t]*B + [0]*B (reference L900-901)
        let t = timestep.asType(hidden.dtype)
        let temb = timeTextEmbed(concatenated([t, t * 0], axis: 0), hidden)

        let imageRotaryEmb = posEmbed(
            videoFHW: imgShapes.map { (frame: $0.0, height: $0.1, width: $0.2) },
            txtSeqLen: txtLen)

        // Per-token index: 0 for the target grid, 1 for ALL conditioning tokens
        // (reference L902-906).
        let targetTokens = imgShapes[0].0 * imgShapes[0].1 * imgShapes[0].2
        let condTokens = imgShapes.dropFirst().reduce(0) { $0 + $1.0 * $1.1 * $1.2 }
        precondition(
            targetTokens + condTokens == imgLen,
            "img token count \(imgLen) != target \(targetTokens) + cond \(condTokens)")
        let modulateIndex: MLXArray? = condTokens > 0
            ? broadcast(
                concatenated(
                    [MLXArray.zeros([targetTokens], type: Int32.self),
                     MLXArray.ones([condTokens], type: Int32.self)]
                )[.newAxis, 0...],
                to: [batch, imgLen])
            : nil

        // Additive joint mask in [text, image] order; nil when the text mask is full.
        let attentionMask = encoderHiddenStatesMask.map {
            Self.buildJointAttentionMask(textMask: $0, imgLen: imgLen).asType(hidden.dtype)
        }

        for block in transformerBlocks {
            (encoder, hidden) = block(
                hiddenStates: hidden, encoderHiddenStates: encoder, temb: temb,
                imageRotaryEmb: imageRotaryEmb, attentionMask: attentionMask,
                modulateIndex: modulateIndex)
        }

        // norm_out modulates with the real-t temb only (reference L969-970).
        hidden = normOut(hidden, conditioning: temb[..<batch])
        return projOut(hidden)
    }

    /// Additive joint mask [B, 1, 1, S_txt + L_img]; -inf on padded text positions.
    static func buildJointAttentionMask(textMask: MLXArray, imgLen: Int) -> MLXArray {
        let bsz = textMask.dim(0)
        let imgOnes = MLXArray.ones([bsz, imgLen], dtype: .bool)
        let joint = concatenated([textMask.asType(.bool), imgOnes], axis: 1)
        let additive = MLX.which(joint, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(.float32)
        return additive[0..., .newAxis, .newAxis, 0...]
    }
}
