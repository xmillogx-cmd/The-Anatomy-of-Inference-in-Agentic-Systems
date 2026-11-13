"""Tests for inference_math.py — reference values computed by hand from headers.

Run:  python tests\test_inference_math.py -v   (from repo root; CPU-only, stdlib)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inference_math as im


class TestInferenceMath(unittest.TestCase):
    def setUp(self):
        self.models = {k: v["arch"] for k, v in im.REFERENCE_MODELS.items()}
        self.gpu4090 = im.REFERENCE_GPUS["RTX-4090"]

    # --- Block 5: parameter formulas and footprint =============================
    def test_param_count_vs_file_size(self):
        """Computed params x dtype must match file size within <=1%."""
        for name, ref in im.REFERENCE_MODELS.items():
            if ref.get("text_only") or ref["file_mb"] is None:
                continue  # gemma-4-e2b: multimodal file, check N/A
            a = ref["arch"]
            expected = im.footprint_bytes(a) + ref.get("extra_file_bytes_u8", 0)
            actual = ref["file_mb"] * 2**20
            self.assertLess(abs(expected - actual) / actual, 0.01, name)

    def test_gemm_params_hand_values(self):
        """Hand recomputation of gemm_params (layers + lm_head GEMM; tied — without vocab·d)."""
        # qwen2.5-0.5b: per_layer = 896^2*2 + 2*896*128 + 3*896*4864 = 14_909_440;
        # tied lm_head -> gemm = 24 * per_layer (embedding table not read)
        self.assertEqual(im.per_layer_gemm_params(self.models["qwen2.5-0.5b"]), 14_909_440)
        self.assertEqual(im.gemm_params(self.models["qwen2.5-0.5b"]), 357_826_560)
        # qwen3-4b: per_layer = 2560*4096 + 2*2560*1024 + 4096*2560 + 3*2560*9728
        self.assertEqual(im.per_layer_gemm_params(self.models["qwen3-4b"]), 100_925_440)
        self.assertEqual(im.gemm_params(self.models["qwen3-4b"]), 3_633_315_840)

    def test_gemma_hetero_arch_values(self):
        """Gemma-4: heterogeneous layers + SWA KV (ground truth from GGUF v3 tensor list; confirmed by E1 fits)."""
        a = self.models["gemma-4-12b-it"]
        # 40 local SWA (224_133_120) + 8 global full layers (241_827_840 — no attn_v, V=K);
        # tied lm_head — without vocab·d. From tensor list: 40×224_133_120 + 8×241_827_840 = 10_899_947_520
        self.assertEqual(im.gemm_params(a), 10_899_947_520)
        # KV f16 S=32k: only the 8 global layers grow linearly (32768*8_192 + min(32768,1024)*163_840)*2 — SWA bounded by window
        self.assertEqual(im.kv_cache_bytes(a, 32768, bits_kv=16), 872_415_232)
        # S=512 < window: both layer types linear (min(S, window)=S) — same sum
        self.assertEqual(im.kv_cache_bytes(a, 512, bits_kv=16), 176_160_768)
        # q8 KV (bits_kv=8 -> e=1): (32768*8_192 + 1024*163_840)*1 — half of f16 at the same length
        self.assertEqual(im.kv_cache_bytes(a, 32768, bits_kv=8), 436_207_616)
        # S* (w=4.95 bits, kv=f16): linear part only (global layers); SWA — constant, ≈ 412k
        s_star = im.kv_crossover_tokens(a, bits_w=4.95, bits_kv=16)
        self.assertTrue(410_000 < s_star < 415_000, f"S* = {s_star}")

    # --- Block 3: KV cache =====================================================
    def test_kv_exact_values(self):
        """Exact KV values (2*(K+V)*L*kv_dim*S*e)."""
        self.assertEqual(im.kv_cache_bytes(self.models["qwen2.5-3b"], 4096), 150_994_944)
        self.assertEqual(im.kv_cache_bytes(self.models["tinyllama-1.1b"], 8192), 184_549_376)
        # FP8 KV (e=1): qwen2.5-0.5b @32k = 2*24*128*32768*1
        self.assertEqual(im.kv_cache_bytes(self.models["qwen2.5-0.5b"], 32768, bits_kv=8), 201_326_592)

    def test_gqa_halves_kv_vs_mha(self):
        """GQA: kv_dim < q_dim -> smaller KV; qwen3-4b (kv=q/4) is 4x less than MHA."""
        a = self.models["qwen3-4b"]
        mha = im.Arch("mha", a.d_model, a.n_layers, a.q_dim, a.q_dim, a.inter,
                      "swiglu", a.vocab, a.tied, a.dtype_bytes)
        ratio = im.kv_cache_bytes(mha, 1024) / im.kv_cache_bytes(a, 1024)
        self.assertAlmostEqual(ratio, a.q_dim / a.kv_dim)

    # --- Blocks 1-2: roofline and TPS ceiling =============================
    def test_decode_ai_law(self):
        """Law of AI decode: at S=0 and native dtype AI == 16/dtype_bits (smollm F32 -> 0.5)."""
        a = self.models["smollm-135m"]   # the only F32 fixture
        self.assertAlmostEqual(im.ai_decode(a), 16 / a.dtype_bits, places=9)
        self.assertLess(abs(im.ai_decode(a) - 0.5), 1e-9)

    def test_decode_memory_bound_all_gpus(self):
        """AI(S=0) < ridge of every GPU -> decode is memory-bound for all fixtures."""
        min_ridge = min(g["ridge"] for g in im.REFERENCE_GPUS.values())
        for name, ref in im.REFERENCE_MODELS.items():
            self.assertLess(im.ai_decode(ref["arch"]), min_ridge, name)

    def test_tps_ceiling_hand_values(self):
        """qwen2.5-0.5b @1008 GB/s (RTX-4090), S=0: bytes = P_gemm*2B = 715_653_120 -> ~1408.5 tps."""
        a = self.models["qwen2.5-0.5b"]
        bw = self.gpu4090["bw_gbps"]
        tps_fp16 = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0))
        tps_int4 = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0, bits_w=4))
        self.assertTrue(1390 < tps_fp16 < 1430, f"tps_fp16 = {tps_fp16}")   # ~1408.5
        self.assertTrue(5600 < tps_int4 < 5670, f"tps_int4 = {tps_int4}")    # ~5634.0

    def test_quant_factor_and_tps_ratio(self):
        """Ideal quantization gain is linear in read volume: int4/fp16 TPS ratio == quant_factor(16,4) == 4."""
        a = self.models["qwen2.5-0.5b"]
        bw = self.gpu4090["bw_gbps"]
        tps16 = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0))
        tps4 = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0, bits_w=4))
        self.assertEqual(im.quant_factor(16, 4), 4)
        self.assertAlmostEqual(tps4 / tps16, 4.0, places=9)

    def test_kv_crossover_hand_values(self):
        """S* = W_read/slope: qwen2.5-0.5b ~58_240 (tied lm_head: W smaller), qwen3-4b ~49_280."""
        s1 = im.kv_crossover_tokens(self.models["qwen2.5-0.5b"])
        s2 = im.kv_crossover_tokens(self.models["qwen3-4b"])
        self.assertTrue(57_000 < s1 < 60_000, f"S* qwen2.5-0.5b = {s1}")
        self.assertTrue(48_000 < s2 < 51_000, f"S* qwen3-4b = {s2}")

    def test_ai_prefill_compute_bound(self):
        """Prefill AI grows with T: @4096 ~3827 FLOP/byte — far above 30x ridge (compute-bound)."""
        a = self.models["qwen2.5-0.5b"]
        ai = im.ai_prefill(a, 4096)
        self.assertGreater(ai, 30 * self.gpu4090["ridge"])
        self.assertLess(ai, 4096)

    # --- Prefill: exact FLOPs and ceiling (E2/E3) ==========================
    def test_prefill_flops_hand_values(self):
        """Gemma-4 @131k hand calc: GEMM = s*2*P_gemm; full attn 8 layers O(s^2); SWA 40 layers O(s*window)."""
        a = self.models["gemma-4-12b-it"]
        s = 131_072
        # GEMM: 131_072 * 2 * 10_899_947_520 (per-layer table sum, tied lm_head)
        gemm = s * 2 * im.gemm_params(a)    # 2_857_355_842_682_880
        self.assertEqual(gemm, 2_857_355_842_682_880)
        # Full layers (window=0): 8 * 4*8192 * s*(s-1)/2 — quadratic part
        full = a.n_layers_full_attn * 4 * a.q_dim_full_layers * s * (s - 1) // 2   # 2_251_782_633_816_064
        self.assertEqual(full, 2_251_782_633_816_064)
        # SWA layers: window*(window-1)/2 + (s-window)*window, window=1024 — linear part
        w = a.swa_window
        n_swa = a.n_layers - a.n_layers_full_attn
        swa = n_swa * 4 * a.q_dim_swa_layers * (w * (w - 1) // 2 + (s - w) * w)    # 87_616_997_294_080
        self.assertEqual(swa, 87_616_997_294_080)
        # Exact total pin — theory anchor for E2/E3 predictions
        self.assertEqual(im.prefill_flops(a, s), 5_196_755_473_793_024)
        # S=1: no position pairs -> attention core = 0, FLOPs == 2*P_gemm
        self.assertEqual(im.prefill_flops(a, 1), 2 * im.gemm_params(a))

    def test_prefill_flops_homogeneous_hand_value(self):
        """qwen2.5-0.5b @4096 (by hand): GEMM 2_931_315_179_520 + attn 721_378_344_960."""
        a = self.models["qwen2.5-0.5b"]
        s = 4096
        gemm = s * 2 * im.gemm_params(a)                     # 2_931_315_179_520
        attn = a.n_layers * 4 * a.q_dim * s * (s - 1) // 2   # 721_378_344_960
        self.assertEqual(gemm + attn, im.prefill_flops(a, s))          # exact total
        self.assertEqual(im.prefill_flops(a, s), 3_652_693_524_480)

    def test_tps_prefill_ceiling_s1_equals_compute(self):
        """Prefill ceiling at S=1 == TFLOPS / (2*P_gemm FLOPs/token): 100 TF / 715.6M ~ 139.8k t/s."""
        a = self.models["qwen2.5-0.5b"]
        tps = im.tps_prefill_ceiling(100.0, a, 1)
        self.assertTrue(139_000 < tps < 140_000, f"tps = {tps}")

    # --- Block 6: multi-GPU splits — bus model (C-20) ======================
    def test_split_gpu_bytes_hand_values(self):
        """gemma-4 per-card bytes = (W + KV(S))/2; W = P_gemm*0.61776B ~= 6_733_551_580 B."""
        a = self.models["gemma-4-12b-it"]
        # W: 10_899_947_520 * 0.61776 (Q4_K_M effective file bytes/param)
        w = im.gemm_params(a) * a.dtype_bytes
        self.assertAlmostEqual(w, 6_733_551_579.9552, delta=1e-3)
        # KV f16 @131k: (131_072*8_192 + min(131_072,1024)*163_840)*2 — SWA bounded by window
        kv = im.kv_cache_bytes(a, 131_072, bits_kv=16)
        self.assertEqual(kv, (131_072 * 8_192 + 1024 * 163_840) * 2)
        # Per-card pins @131k/262k x {f16,q8} — mirror-KV ceiling table (E3 v3 predictions)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 131_072, bits_kv=16), 4_608_289_773.9776, delta=1)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 131_072, bits_kv=8), 3_987_532_781.9776, delta=1)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 262_144, bits_kv=16), 5_682_031_597.9776, delta=1)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 262_144, bits_kv=8), 4_524_403_693.9776, delta=1)

    def test_ts_ceiling_double_sg(self):
        """Identity: TS ceiling (N cards read in parallel) == SG ceiling at per-card bytes; ratio to SG == N."""
        a = self.models["qwen2.5-0.5b"]
        bw = 397.7   # E1 measured BW_eff (CUDA)
        s = 32_768
        sg_bytes = im.bytes_read_per_token(a, s)
        ts_tps = im.tps_ceiling_ts(bw, a, s, n_devices=2)
        self.assertAlmostEqual(ts_tps, im.tps_ceiling(bw, im.split_gpu_bytes_per_device(a, s)), places=6)
        ratio = ts_tps / im.tps_ceiling(bw, sg_bytes)
        self.assertAlmostEqual(ratio, 2.0, delta=1e-9)   # TS/SG == N at bw=397.7

    def test_ts_comm_hand_value(self):
        """allreduce per token: L*2*d_model*e_act = 48*2*3840*2 B (PCIe; secondary term)."""
        a = self.models["gemma-4-12b-it"]
        self.assertEqual(im.ts_comm_bytes_per_token(a), 737_280)

    def test_split_homogeneous_exact(self):
        """qwen2.5-0.5b @1k f16, N=2: (W + KV)/2 = (715_653_120 + 12_582_912)/2 — exact."""
        a = self.models["qwen2.5-0.5b"]
        w_bytes = im.gemm_params(a) * a.dtype_bytes    # P_gemm*2B (BF16) -> 715_653_120
        kv_bytes = im.kv_cache_bytes(a, 1024)          # 2*24*128*1024*2 -> 12_582_912
        self.assertEqual(w_bytes + kv_bytes, 715_653_120 + 12_582_912)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 1024), (715_653_120 + 12_582_912) / 2, delta=1e-9)

    # --- Per-layer model: heterogeneous layers of gemma-4 ==================
    def test_layer_gemm_params_i_hand_values(self):
        """SWA layer (il=0) vs global layer (il=5): 224_133_120 / 241_827_840."""
        a = self.models["gemma-4-12b-it"]
        # SWA: q/k/v/o + FFN; global: attn_v ABSENT (V=K) -> larger P_layer, no kv_proj read
        self.assertEqual(im.layer_gemm_params_i(a, 0), 224_133_120)
        self.assertEqual(im.layer_gemm_params_i(a, 5), 241_827_840)
        self.assertTrue(im.is_full_attn_layer(a, 5))
        self.assertFalse(im.is_full_attn_layer(a, 0))

    def test_kv_growth_terms_hand_values(self):
        """Slope = full layers only (SWA constant after window fill); exactly the E1 fit slope (x1.00)."""
        a = self.models["gemma-4-12b-it"]
        # kv_slope: 8_192 elems/pos * e — f16: 16_384 B/pos, q8: 8_192 B/pos
        self.assertEqual(im.kv_slope_bytes_per_pos(a, bits_kv=16), 16_384)
        self.assertEqual(im.kv_slope_bytes_per_pos(a, bits_kv=8), 8_192)
        # SWA constant: window * kv_elems_swa * e = 1024*163_840*2 (approx. 320 MiB)
        self.assertEqual(im.kv_swa_const_bytes(a, bits_kv=16), 335_544_320)

    def test_layer_kv_read_hand_values(self):
        """KV read per token in one layer: full — all S positions; SWA — window only (same at 131k and 262k)."""
        a = self.models["gemma-4-12b-it"]
        # il=5 (full): S * (8_192/8) * e = 131_072*1024*2 — per-layer global slope, exact power of two
        self.assertEqual(im.layer_kv_read_bytes_per_token(a, 5, 131_072, bits_kv=16), 268_435_456)
        # il=0 (SWA): min(S,1024) * (163_840/40) * e = 1024*4096*2 — independent of S past the window
        self.assertEqual(im.layer_kv_read_bytes_per_token(a, 0, 131_072, bits_kv=16), 8_388_608)
        self.assertEqual(im.layer_kv_read_bytes_per_token(a, 0, 262_144, bits_kv=16), 8_388_608)

    # --- TS modes: ideal vs mirror-KV patch (E3 v3 predictions) ============
    def test_ts_mirrored_hand_values(self):
        """mirror-KV bytes/card = W/2 + SWA_KV(S)/2 + GLOBAL_KV(S) full read (not split)."""
        a = self.models["gemma-4-12b-it"]
        # @131k f16: 10_899_947_520*0.61776/2 + 335_544_320/2 + 131_072*16_384
        expected = (im.gemm_params(a) * a.dtype_bytes / 2
                    + im.kv_swa_const_bytes(a, bits_kv=16) / 2
                    + 131_072 * im.kv_slope_bytes_per_pos(a, bits_kv=16))
        self.assertAlmostEqual(im.ts_mode_bytes_mirrored_single_head_kv(a, 131_072, bits_kv=16),
                               expected, delta=1e-3)
        # E3 v3 ceiling predictions: BW_eff=397.7 GB/s -> ~70.0 t/s @131k f16
        self.assertTrue(69.5 < im.tps_ceiling_ts_mirrored(397.7, a, 131_072, bits_kv=16) < 70.5)
        # @262k: ~50.8 t/s (compare with C-18 PP 33.19/26.89 — TS should win by parallel reads)
        self.assertTrue(50.0 < im.tps_ceiling_ts_mirrored(397.7, a, 262_144, bits_kv=16) < 51.5)

    def test_ts_mirrored_equals_ideal_double_context(self):
        """Identity: mirror at context S == head-split (ideal) at 2S — same bytes per card."""
        a = self.models["gemma-4-12b-it"]
        for s in (131_072, 262_144):
            for kv in (16, 8):
                mirrored = im.ts_mode_bytes_mirrored_single_head_kv(a, s, bits_kv=kv)
                ideal_2s = im.split_gpu_bytes_per_device(a, 2 * s, bits_kv=kv)
                self.assertAlmostEqual(mirrored, ideal_2s, delta=1e-3, msg=f"s={s} kv={kv}")

    def test_pp_and_ideal_modes_equal(self):
        """PP and TS with head-split KV give identical per-card bytes (temporal difference only)."""
        a = self.models["gemma-4-12b-it"]
        for s in (131_072, 262_144):
            for kv in (16, 8):
                self.assertEqual(im.pp_mode_bytes_per_device(a, s, bits_kv=kv),
                                 im.ts_mode_bytes_ideal(a, s, bits_kv=kv))

    def test_ts_mirrored_homogeneous_equals_ideal(self):
        """Homogeneous arch: no single-head layers -> the patch is not needed (falls back to ideal)."""
        a = self.models["qwen2.5-0.5b"]
        for s in (1024, 32_768):
            self.assertEqual(im.ts_mode_bytes_mirrored_single_head_kv(a, s),
                             im.ts_mode_bytes_ideal(a, s))


if __name__ == "__main__":
    unittest.main(verbosity=2)
