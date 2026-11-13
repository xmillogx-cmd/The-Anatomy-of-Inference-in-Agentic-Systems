"""AoI — math core of the "Physics of LLM Inference" (roadmap blocks 1–3, 5).

CPU-only, stdlib. All architecture parameters are ground truth from disk:
headers of local safetensors fixtures (catalog set by env var AOI_MODELS_DIR;
see tools/parse_headers.py). Bus bandwidth and GPU ridge points — public
datasheet values (nvidia.com), conservative peaks.

Ideal decode step model (batch=1): one pass over all active weight GEMM matrices
+ KV cache read up to the current position S.
Approximations: embedding row read O(d·e) and norms/biases (<0.5% for most models)
are not counted; attention softmax FLOPs O(S·d) — a secondary term relative to matmuls
(not included in flops_per_token).

Run:  python inference_math.py report   — AI/TPS/KV/crossover table.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    """Transformer architecture, parameters from the safetensors header (ground truth).

    q_dim  = n_heads * head_dim      (size of q_proj output / o_proj input)
    kv_dim = n_kv_heads * head_dim   (GQA-aware: take from k/v_proj shape)
    """
    name: str
    d_model: int
    n_layers: int
    q_dim: int
    kv_dim: int
    inter: int
    mlp_kind: str      # swiglu | geglu | relu | gelu  (swiglu/geglu -> 3 matrices, else 2)
    vocab: int
    tied: bool         # lm_head shares the table with embed_tokens
    dtype_bytes: float # size of a weight element in file bytes (BF16/F16=2.0, F32=4.0)
    ple_table_params: int = 0      # extra embedding tables (Gemma PLE), parameter count
    per_layer_extra_gemm: int = 0  # extra GEMM matrices per layer (PLE gate/proj), params
    layer_gemm_params: tuple[int, ...] | None = None  # heterogeneous layers: GEMM params of each layer
    kv_elems_full: int | None = None   # Σ over full-attention layers of 2·kv_dim_i (K+V elems/token)
    kv_elems_swa: int | None = None    # same for SWA layers; buffer bounded by swa_window tokens
    swa_window: int = 0                # sliding attention window in tokens (0 — no SWA)
    n_layers_full_attn: int | None = None  # number of full-attention layers (heterogeneous arch.; None → all layers full)
    q_dim_full_layers: int | None = None   # h_q·d_head of full layers (gemma-4: attn_q [3840x8192] -> 8192)
    q_dim_swa_layers: int | None = None    # h_q·d_head of SWA layers (gemma-4: attn_q [3840x4096] -> 4096)
    full_attn_layers: tuple[int, ...] | None = None  # heterogeneous arch.: indices of full-attention layers (None → homogeneous)

    @property
    def dtype_bits(self) -> float:
        return self.dtype_bytes * 8


def mlp_params(a: Arch) -> int:
    """MLP params of one layer: 3·d·inter for the GLU family, else 2·d·inter."""
    n = 3 if a.mlp_kind in ("swiglu", "geglu") else 2
    return n * a.d_model * a.inter


def per_layer_gemm_params(a: Arch) -> int:
    """Total params of all GEMM matrices of one layer (weights, without bias/norms)."""
    return (a.d_model * a.q_dim            # q_proj
            + 2 * a.d_model * a.kv_dim     # k_proj + v_proj
            + a.q_dim * a.d_model          # o_proj
            + mlp_params(a)                # MLP
            + a.per_layer_extra_gemm)


def gemm_params(a: Arch) -> int:
    """All matrices read per decode token (batch=1): layers + lm_head GEMM.

    With tied lm_head the embedding table is shared — decode reads one row O(d),
    so vocab·d does NOT enter the read volume.
    """
    if a.layer_gemm_params is not None:
        layers = sum(a.layer_gemm_params)
    else:
        layers = a.n_layers * per_layer_gemm_params(a)
    return layers + (0 if a.tied else a.vocab * a.d_model)


def footprint_bytes(a: Arch, bits_w: float | None = None) -> float:
    """VRAM occupation of weights. Norms/biases (<0.5% for most models) not counted."""
    b = bits_w or a.dtype_bits
    emb_tables = 1 if a.tied else 2
    if a.layer_gemm_params is not None:
        layers = sum(a.layer_gemm_params)
    else:
        layers = a.n_layers * per_layer_gemm_params(a)
    params = (layers + a.vocab * a.d_model * emb_tables + a.ple_table_params)
    return params * b / 8


def kv_cache_bytes(a: Arch, seq_len: int, bits_kv: float | None = None) -> float:
    """K+V across all layers at context length seq_len (batch=1), bytes.

    Heterogeneous layers (gemma-4): full attention is linear in S; SWA layers —
    ring buffer bounded by the window: min(S, swa_window) tokens.
    """
    e = (bits_kv or a.dtype_bits) / 8
    if a.kv_elems_full is not None:
        elems = seq_len * a.kv_elems_full
        if a.kv_elems_swa:
            elems += min(seq_len, a.swa_window) * a.kv_elems_swa
        return elems * e
    return 2 * a.n_layers * a.kv_dim * seq_len * e


def bytes_read_per_token(a: Arch, seq_len: int, bits_w=None, bits_kv=None) -> float:
    """Bytes per decode token: weights (one pass) + KV read up to position seq_len."""
    b = bits_w or a.dtype_bits
    return gemm_params(a) * b / 8 + kv_cache_bytes(a, seq_len, bits_kv)


def tps_ceiling(bw_gbps: float, bytes_read: float) -> float:
    """TPS ceiling (batch=1): BW / read volume per token. Memory-bound model."""
    return bw_gbps * 1e9 / bytes_read


def flops_per_token(a: Arch) -> float:
    """Matmul FLOPs of one decode step (~2·GEMM params)."""
    return 2.0 * gemm_params(a)


def _attn_core_flops(qdim: int, tokens: int, window: int = 0) -> float:
    """Attention core FLOPs (QK^T + attn@V) for `tokens` prefill in one layer.

    Position p reads min(p, window) keys; per position — 4·(h_q·d_head)·min(p,W)
    FLOP (2× dot products QK^T + 2× attn@V). GQA does not change the op count.
    """
    S = max(tokens, 1)
    if window <= 0 or S <= window:
        s = S * (S - 1) / 2.0
    else:
        s = window * (window - 1) / 2.0 + (S - window) * window
    return 4.0 * qdim * s


def prefill_flops(a: Arch, tokens: int) -> float:
    """Total FLOPs of a full `tokens`-token prefill (batch=1).

    GEMM part S·2·P_gemm (each token = one pass over weights); attention core
    O(S²) on full layers, O(S·window) on SWA. Softmax O(S·d) — secondary term,
    not included (as in flops_per_token).
    """
    S = max(tokens, 1)
    flops = S * 2.0 * gemm_params(a)
    if a.n_layers_full_attn is not None:
        n_swa = a.n_layers - a.n_layers_full_attn
        flops += a.n_layers_full_attn * _attn_core_flops(a.q_dim_full_layers, S, 0)
        if n_swa > 0:
            flops += n_swa * _attn_core_flops(a.q_dim_swa_layers, S, a.swa_window)
    else:
        flops += a.n_layers * _attn_core_flops(a.q_dim, S, a.swa_window)
    return flops


def tps_prefill_ceiling(tflops_eff: float, a: Arch, tokens: int) -> float:
    """Prefill TPS ceiling (compute-bound): effective TFLOPS / FLOPs per token."""
    S = max(tokens, 1)
    return tflops_eff * 1e12 / (prefill_flops(a, S) / S)


def ai_decode(a: Arch, seq_len=0, bits_w=None, bits_kv=None) -> float:
    """Arithmetic Intensity of decode (FLOP/byte).

    Law: at S=0 and native dtype  AI == 16/dtype_bits
    (BF16/F16 -> 1.0, F32 -> 0.5) — always below the ridge point of any GPU.
    """
    return flops_per_token(a) / bytes_read_per_token(a, seq_len, bits_w, bits_kv)


def ai_prefill(a: Arch, tokens: int, bits_w=None) -> float:
    """AI of prefill (batch=1): FLOPs grow with T, weights are read once.

    Reads during prefill: weights + KV write for T tokens. At large T AI ~ T·16/bits_w.
    """
    b = bits_w or a.dtype_bits
    reads = gemm_params(a) * b / 8 + kv_cache_bytes(a, tokens, bits_w)
    return tokens * flops_per_token(a) / reads


def quant_factor(bits_from: float, bits_to: float) -> float:
    """Ideal TPS gain from weight quantization (linear in read volume)."""
    return bits_from / bits_to


def kv_crossover_tokens(a: Arch, bits_w=None, bits_kv=None) -> float:
    """Context length S* where KV read == weight read.

    Above S*, quantizing the KV starts to give a bigger effect than quantizing weights.
    """
    b = bits_w or a.dtype_bits
    if a.kv_elems_full is not None:
        slope = a.kv_elems_full * (bits_kv or b) / 8   # linear part only; SWA — constant
    else:
        slope = 2 * a.n_layers * a.kv_dim * (bits_kv or b) / 8
    return gemm_params(a) * b / 8 / slope


# --- Block 6: multi-GPU splits — per-device bus model ========

def split_gpu_bytes_per_device(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Bytes of VRAM read per decode token (batch=1) on ONE device when split.

    In the ideal both modes give each card P_gemm·b/8/N of weights + KV(S)/N: PP splits
    layers together with their KV, TS splits matrices — in current llama.cpp the KV cache is also
    distributed across devices (meta-callback pattern_kv_cache →
    GGML_BACKEND_SPLIT_AXIS_0; K/V tensor shape [n_embd_k_gqa, kv_size], i.e.
    axis 0 = kv-head space — checked against master on 24.10.2026). The difference
    between modes is temporal: PP at batch=1 serializes the halves of a token (no TPS gain
    — C-18), TS parallelizes them with allreduce overhead over PCIe. For gemma-4
    the layer split is symmetric (24/24, 4 full layers in each half) ⇒ formula exact.
    """
    b = bits_w or a.dtype_bits
    return (gemm_params(a) * b / 8 + kv_cache_bytes(a, seq_len, bits_kv)) / n_devices


def tps_ceiling_ts(bw_gbps: float, a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Ideal decode TPS ceiling under tensor split (TS): N cards read in parallel.

    Upper bound only — allreduce/synchronization overhead not accounted for. For PP at batch=1 the chain is serialized ⇒ real gain ~0 (C-18); the formula does not apply to PP as a TPS estimate (only as "what would happen with parallel reads").
    """
    b = bits_w or a.dtype_bits
    total = gemm_params(a) * b / 8 + kv_cache_bytes(a, seq_len, bits_kv)
    return n_devices * bw_gbps * 1e9 / total


def ts_comm_bytes_per_token(a: Arch, act_dtype_bytes: float = 2.0) -> float:
    """TS (N=2): bytes over PCIe per token — allreduce of the partial hidden state [d_model]
    at each layer (send+receive). Secondary term relative to VRAM reads
    (~0.74 MB/token vs ~9 GB @131k f16 for gemma-4); the PP analog is inter-card activation transfer at layer boundaries of the same order, but without allreduce.
    """
    return a.n_layers * 2 * a.d_model * act_dtype_bytes


def is_full_attn_layer(a: Arch, il: int) -> bool:
    """Is the layer with index il full attention? Heterogeneous arch (gemma-4): by table;
    homogeneous: all layers full."""
    if a.full_attn_layers is None:
        return True
    return il in a.full_attn_layers


def layer_gemm_params_i(a: Arch, il: int) -> int:
    """GEMM matrix params of one layer (heterogeneous arch — from the per-layer table)."""
    if a.layer_gemm_params is not None:
        return a.layer_gemm_params[il]
    return per_layer_gemm_params(a)


def layer_weight_bytes_per_token(a: Arch, il: int, bits_w=None) -> float:
    """Bytes of weight VRAM read for layer il per decode token (one pass over matrices)."""
    b = bits_w or a.dtype_bits
    return layer_gemm_params_i(a, il) * b / 8


def layer_flops_per_token(a: Arch, il: int) -> float:
    """Matmul FLOPs of the decode step in layer il (~2·P_layer)."""
    return 2.0 * layer_gemm_params_i(a, il)


def _kv_elems_per_full_layer(a: Arch) -> int:
    """K+V elems/token on a full-attention layer (uniformity within a layer type —
    true for gemma-4: 8·1024/8 = 1024; homogeneous arch → 2·kv_dim)."""
    if a.kv_elems_full is None or not a.n_layers_full_attn:
        return 2 * a.kv_dim
    return a.kv_elems_full // a.n_layers_full_attn


def _kv_elems_per_swa_layer(a: Arch) -> int:
    """K+V elems/token on an SWA layer (gemma-4: 163_840/40 = 4096; no SWA → 0)."""
    if a.kv_elems_swa is None or not a.swa_window:
        return 0
    n_swa = a.n_layers - (a.n_layers_full_attn or 0)
    return a.kv_elems_swa // max(n_swa, 1)


def layer_kv_read_bytes_per_token(a: Arch, il: int, seq_len: int, bits_kv=None) -> float:
    """Bytes of KV cache read in layer il per decode token at position S.

    A full-attention layer reads all S positions; SWA — a ring buffer bounded by the window:
    min(S, swa_window). This is exactly "context growth → bus byte growth":
    ∂/∂S for full layers = 2·kv_dim_i·e (gemma-4 global: 16_384 B/pos over the whole model),
    for SWA after window fill — 0.
    """
    e = (bits_kv or a.dtype_bits) / 8
    if is_full_attn_layer(a, il):
        return seq_len * _kv_elems_per_full_layer(a) * e
    w = min(seq_len, a.swa_window)
    return w * _kv_elems_per_swa_layer(a) * e


def kv_slope_bytes_per_pos(a: Arch, bits_kv=None) -> float:
    """∂KV/∂S — bytes added to the KV cache read at each new position (decode).

    Homogeneous arch: 2·L·kv_dim·e. Heterogeneous (gemma-4): only full layers grow linearly → kv_elems_full·e; SWA buffer is constant after window fill.
    gemma-4 f16: 8_192·2 = 16_384 B/pos — exactly the slope of the E1 fit (×1.00, R²=0.998).
    """
    e = (bits_kv or a.dtype_bits) / 8
    if a.kv_elems_full is not None:
        return a.kv_elems_full * e
    return 2 * a.n_layers * a.kv_dim * e


def kv_swa_const_bytes(a: Arch, bits_kv=None) -> float:
    """Constant SWA term for S ≥ window: ring buffer of swa_window tokens.

    Independent of S — context growth past the window does not increase SWA cache reads.
    gemma-4 f16: 1024·163_840·2 = 335_544_320 B (≈320 MiB).
    """
    e = (bits_kv or a.dtype_bits) / 8
    if a.kv_elems_swa is None:
        return 0.0
    return a.swa_window * a.kv_elems_swa * e


def pp_mode_bytes_per_device(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Layer split (PP): bytes of VRAM read per token on ONE card = (W + KV(S))/N.

    PP splits layers together with their KV — each half holds its own weights and the KV of its layers; at batch=1 the chain between halves is serialized ⇒ no TPS gain (C-18), the formula gives only bytes per card, not an acceleration ceiling.
    """
    return split_gpu_bytes_per_device(a, seq_len, n_devices, bits_w, bits_kv)


def ts_mode_bytes_ideal(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """TS with head-split KV in ALL layers (works when n_kv_heads ≥ N in each layer): bytes per card = (W + KV(S))/N — same formula as PP/ideal; the ceiling is limited by allreduce (ts_comm_bytes_per_token), not by VRAM reads."""
    return split_gpu_bytes_per_device(a, seq_len, n_devices, bits_w, bits_kv)


def ts_mode_bytes_mirrored_single_head_kv(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """TS with the mirror-KV patch for layers where kv_heads < N (gemma-4 global: 1 kv-head).

    Mechanism (source gemma4.cpp + ggml-backend-meta.cpp): GEMM matrices q/k/o and FFN are cut by heads/rows; SWA KV is head-split (/N — 8 kv_heads ≥ 2 devices, works in current master too); the full-attention KV cache is MIRRORED — each card reads ALL S positions (head split impossible: 1 < N), per-row qk-norm/rms(V) run on MIRRORED tensors, RoPE as well.

    bytes/card = W/N + min(S,W)·kv_elems_swa·e/N + S·kv_elems_full·e (full — NOT split). Homogeneous archs without single-head layers: the patch is not needed → ideal formula.
    """
    if a.kv_elems_full is None or a.kv_elems_swa is None:
        return ts_mode_bytes_ideal(a, seq_len, n_devices, bits_w, bits_kv)
    b = bits_w or a.dtype_bits
    e = (bits_kv or b) / 8
    w = gemm_params(a) * b / 8 / n_devices
    kv_swa = min(seq_len, a.swa_window) * a.kv_elems_swa * e / n_devices   # head-split
    kv_full = seq_len * a.kv_elems_full * e                                # mirror — full read on each card
    return w + kv_swa + kv_full


def tps_ceiling_ts_mirrored(bw_gbps: float, a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Decode TPS ceiling for TS with mirror-KV (patched mode): BW / bytes per card.

    Upper bound — allreduce/synchronization overhead not accounted for. E3 v3 predictions (BW_eff=397.7 GB/s, two RTX 5060 Ti): gemma-4 @131k f16 ≈ 70.0 t/s, @262k ≈ 50.8; compare with C-18 PP (33.19/26.89) — TS should win thanks to parallel reads.
    """
    return bw_gbps * 1e9 / ts_mode_bytes_mirrored_single_head_kv(a, seq_len, n_devices, bits_w, bits_kv)


# ---------------------------------------------------------------------------
# Reference tables. Architectures — ground truth from safetensors headers of local fixtures
# (AOI_MODELS_DIR, tools/parse_headers.py); file_mb = total shard size in MiB. GPU — public datasheet values.
# ---------------------------------------------------------------------------

def _m(name, d, l, q, kv, inter, kind, vocab, tied, dt, **kw):
    return Arch(name=name, d_model=d, n_layers=l, q_dim=q, kv_dim=kv, inter=inter,
                mlp_kind=kind, vocab=vocab, tied=tied, dtype_bytes=dt, **kw)


# gemma-4: heterogeneous layer order — full attention at blk.5/11/../47 (step 6),
# SWA everywhere else; the layer_gemm_params table is indexed by real layer numbers.
_GEMMA4_FULL = (5, 11, 17, 23, 29, 35, 41, 47)
_GEMMA4_LAYER_P = tuple(241_827_840 if il in _GEMMA4_FULL else 224_133_120 for il in range(48))


REFERENCE_MODELS: dict[str, dict] = {
    "smollm-135m": {
        "arch": _m("smollm-135m", 576, 30, 576, 192, 1536, "swiglu", 49152, True, 4.0),
        "file_mb": 513.2, "source": "local safetensors header (F32)",
    },
    "qwen2.5-0.5b": {
        "arch": _m("qwen2.5-0.5b", 896, 24, 896, 128, 4864, "swiglu", 151936, True, 2.0),
        "file_mb": 942.3, "source": "local safetensors header (BF16)",
    },
    "tinyllama-1.1b": {
        "arch": _m("tinyllama-1.1b", 2048, 22, 2048, 256, 5632, "swiglu", 32000, False, 2.0),
        "file_mb": 2098.2, "source": "local safetensors header (BF16)",
    },
    "phi-1_5": {
        "arch": _m("phi-1_5", 2048, 24, 2048, 2048, 8192, "relu", 51200, False, 2.0),
        "file_mb": 2705.2, "source": "local safetensors header (F16)",
    },
    "pythia-2.8b": {
        # GPT-NeoX naming; the file contains U8 attention.bias buffers [1,1,2048,2048]
        # per layer (causal mask) — NOT weights, subtract when comparing to file_mb.
        "arch": _m("pythia-2.8b", 2560, 32, 2560, 2560, 10240, "gelu", 50304, False, 2.0),
        "file_mb": 5421.3, "source": "local safetensors header (F16 + U8 bias buffers)",
        "extra_file_bytes_u8": 32 * (2048 * 2048),
    },
    "qwen2.5-3b": {
        "arch": _m("qwen2.5-3b", 2048, 36, 2048, 256, 11008, "swiglu", 151936, True, 2.0),
        "file_mb": 5886.0, "source": "local safetensors header (BF16)",
    },
    "qwen3-4b": {
        "arch": _m("qwen3-4b", 2560, 36, 4096, 1024, 9728, "swiglu", 151936, True, 2.0),
        "file_mb": 7672.3, "source": "local safetensors header (BF16)",
    },
    # Heterogeneous architecture (ground truth from the tensor list + GGUF v3 metadata;
    # confirmed by the E1 measurement on 23.10.2026 — f16 fit slope = theory exactly ×1.00):
    # 40 local SWA layers (sliding_window=1024, kv_dim=2048; blk.0.attn_k [3840x2048])
    # + 8 global full-attention blk.5/11/../47 step 6 (kv_dim=512;
    # blk.5.attn_k [3840x512]). Only the global layers grow linearly in S.
    # P_layer from the tensor list (tmp-src/tensors.txt, analyze_tensors.py): SWA layer = 14
    # tensors / GEMM 224_133_120; GLOBAL layer = **13** tensors — attn_v ABSENT
    # (in global layers V = K, gemma4.cpp: wv ? mm : Kcur) → P_layer(global)=241_827_840.
    # FFN = gate/up/down with GELU (LLM_FFN_GELU in gemma4.cpp; structurally the GLU family,
    # 3 matrices — same bytes/FLOPs). lm_head tied with token_embd — decode reads one
    # row of the table. dtype_bytes = effective file bytes/param of Q4_K_M (~4.94 bits).
    "gemma-4-12b-it": {
        "arch": _m("gemma-4-12b-it", 3840, 48, 4096, 2048, 15360, "geglu", 262144, True, 0.61776,
                   layer_gemm_params=_GEMMA4_LAYER_P,   # order = real layer indices (full at 5/11/../47)
                   kv_elems_full=8 * 2 * 512,       # 8_192 K+V elems/token — global layers (linear in S)
                   kv_elems_swa=40 * 2 * 2048,     # 163_840; buffer bounded by window 1024
                   swa_window=1024,
                   n_layers_full_attn=8,            # full attention: blk.5/11/../47 (step 6)
                   q_dim_full_layers=8192,         # attn_q of global layers [3840x8192]
                   q_dim_swa_layers=4096,          # attn_q of local layers [3840x4096]
                   full_attn_layers=_GEMMA4_FULL),
        "file_mb": 7_381_384_864 / 2**20,           # 7045.9 MiB (GGUF Q4_K_M)
        "source": "local GGUF v3 header + tensor list (Q4_K_M)",
    },
    # Multimodal: the file has vision/audio towers — file_size check N/A;
    # Arch describes only the text transformer (+ PLE table and per-layer PLE GEMM).
    "gemma-4-e2b": {
        "arch": _m("gemma-4-e2b", 1536, 35, 2048, 256, 6144, "geglu", 262144, True, 2.0,
                   ple_table_params=262144 * 8960, per_layer_extra_gemm=2 * 1536 * 256),
        "file_mb": None, "source": "local safetensors header (BF16, text-only arch)",
        "text_only": True,
    },
}

REFERENCE_GPUS: dict[str, dict] = {
    # bw_gbps — peak HBM/GDDR bandwidth; ridge = FLOP/byte
    # (peak dense TFLOPS / BW), conservative per public datasheets.
    "RTX-4090": {"bw_gbps": 1008.0, "ridge": 82.0},   # GDDR6X ~1 TB/s; FP32 82.6 TFLOPS
    "A100-80G": {"bw_gbps": 2039.0, "ridge": 153.0},  # HBM2e 2.039 TB/s; FP16 dense 312 TF
    "H100-SXM": {"bw_gbps": 3350.0, "ridge": 295.0},  # HBM3 3.35 TB/s; BF16 dense ~989 TF
}


def _report() -> None:
    """AI/TPS/KV/crossover table for all models — to seed CONCLUSIONS.md.

    "TPS nat" and the KV column — in the file's native dtype (for BF16/F16 models = fp16;
    smollm F32 and gemma-4 Q4_K_M — effective file bits, not fp16).
    """
    print(f"{'model':<14} {'dt':>3} {'AI(S=0)':>8} {'margin@4090':>12} "
          f"{'TPS nat S=0':>13} {'TPS int4 S=0':>13} {'KV MiB 32k/8k':>15} {'S* cross':>9}")
    for name, ref in REFERENCE_MODELS.items():
        a = ref["arch"]
        ai0 = ai_decode(a)
        m4090 = REFERENCE_GPUS["RTX-4090"]["ridge"] / ai0
        bw = REFERENCE_GPUS["RTX-4090"]["bw_gbps"]
        tps16 = tps_ceiling(bw, bytes_read_per_token(a, 0))
        tps4 = tps_ceiling(bw, bytes_read_per_token(a, 0, bits_w=4))
        kv32k = kv_cache_bytes(a, 32768) / 2**20
        kv8k = kv_cache_bytes(a, 8192) / 2**20
        print(f"{name:<14} {a.dtype_bits:>3.0f} {ai0:>8.3f} {m4090:>12.1f} "
              f"{tps16:>13.1f} {tps4:>13.1f} {kv32k:>7.1f}/{kv8k:<7.1f} "
              f"{kv_crossover_tokens(a):>9.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report()
    else:
        print(__doc__)
