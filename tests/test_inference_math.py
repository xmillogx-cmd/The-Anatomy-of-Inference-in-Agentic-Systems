"""Тесты inference_math.py — эталонные значения посчитаны вручную из header'ов.

Запуск:  python tests\test_inference_math.py -v   (из корня репо; CPU-only, stdlib)
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

    # --- Блок 5: формулы параметров и footprint =============================
    def test_param_count_vs_file_size(self):
        """Считанные параметры x dtype должны совпадать с размером файла <=1%."""
        for name, ref in im.REFERENCE_MODELS.items():
            if ref.get("text_only") or ref["file_mb"] is None:
                continue  # gemma-4-e2b: мультимодальный файл, проверка N/A
            a = ref["arch"]
            expected = im.footprint_bytes(a) + ref.get("extra_file_bytes_u8", 0)
            actual = ref["file_mb"] * 2**20
            self.assertLess(abs(expected - actual) / actual, 0.01, name)

    def test_gemm_params_hand_values(self):
        """Ручной пересчёт gemm_params (слои + lm_head GEMM; tied — без vocab·d)."""
        # qwen2.5-0.5b: per_layer = 896^2*2 + 2*896*128 + 3*896*4864 = 14_909_440;
        # tied lm_head -> gemm = 24 * per_layer (таблица embedding не читается)
        self.assertEqual(im.per_layer_gemm_params(self.models["qwen2.5-0.5b"]), 14_909_440)
        self.assertEqual(im.gemm_params(self.models["qwen2.5-0.5b"]), 357_826_560)
        # qwen3-4b: per_layer = 2560*4096 + 2*2560*1024 + 4096*2560 + 3*2560*9728
        self.assertEqual(im.per_layer_gemm_params(self.models["qwen3-4b"]), 100_925_440)
        self.assertEqual(im.gemm_params(self.models["qwen3-4b"]), 3_633_315_840)

    def test_gemma_hetero_arch_values(self):
        """Gemma-4: гетерогенные слои + SWA-KV (ground truth из tensor list GGUF v3; подтверждено фитами E1)."""
        a = self.models["gemma-4-12b-it"]
        # 40 локальных SWA (224_133_120) + 8 глобальных full-слоёв (241_827_840 — attn_v нет, V=K);
        # tied lm_head — без vocab·d. По tensor list: 40×224_133_120 + 8×241_827_840 = 10_899_947_520
        self.assertEqual(im.gemm_params(a), 10_899_947_520)
        # KV f16 S=32k: линейно растут только 8 глобальных слоёв (32768*8_192 + min(32768,1024)*163_840)*2 — SWA ограничен окном
        self.assertEqual(im.kv_cache_bytes(a, 32768, bits_kv=16), 872_415_232)
        # S=512 < окно: оба типа слоёв линейны (min(S, window)=S) — сумма та же
        self.assertEqual(im.kv_cache_bytes(a, 512, bits_kv=16), 176_160_768)
        # q8 KV (bits_kv=8 -> e=1): (32768*8_192 + 1024*163_840)*1 — вдвое меньше f16 на той же длине
        self.assertEqual(im.kv_cache_bytes(a, 32768, bits_kv=8), 436_207_616)
        # S* (w=4.95 бит, kv=f16): только линейная часть (глобальные слои); SWA — константа, ≈ 412k
        s_star = im.kv_crossover_tokens(a, bits_w=4.95, bits_kv=16)
        self.assertTrue(410_000 < s_star < 415_000, f"S* = {s_star}")

    # --- Блок 3: KV-кэш =====================================================
    def test_kv_exact_values(self):
        """Точные значения KV (2*(K+V)*L*kv_dim*S*e)."""
        self.assertEqual(im.kv_cache_bytes(self.models["qwen2.5-3b"], 4096), 150_994_944)
        self.assertEqual(im.kv_cache_bytes(self.models["tinyllama-1.1b"], 8192), 184_549_376)
        # FP8 KV (e=1): qwen2.5-0.5b @32k = 2*24*128*32768*1
        self.assertEqual(im.kv_cache_bytes(self.models["qwen2.5-0.5b"], 32768, bits_kv=8), 201_326_592)

    def test_gqa_halves_kv_vs_mha(self):
        """GQA: kv_dim < q_dim -> KV меньше; qwen3-4b (kv=q/4) на 4x меньше MHA."""
        a = self.models["qwen3-4b"]
        mha = im.Arch("mha", a.d_model, a.n_layers, a.q_dim, a.q_dim, a.inter,
                      "swiglu", a.vocab, a.tied, a.dtype_bytes)
        ratio = im.kv_cache_bytes(mha, 1024) / im.kv_cache_bytes(a, 1024)
        self.assertAlmostEqual(ratio, a.q_dim / a.kv_dim)

    # --- Блоки 1-2: roofline и потолок TPS ==================================
    def test_decode_ai_law(self):
        """Закон AI decode: при S=0 и нативном dtype AI == 16/dtype_bits."""
        for name, a in self.models.items():
            ai = im.ai_decode(a)
            self.assertAlmostEqual(ai, 16 / a.dtype_bits, delta=1e-6 * (16 / a.dtype_bits), msg=name)
        self.assertAlmostEqual(im.ai_decode(self.models["smollm-135m"]), 0.5)

    def test_decode_memory_bound_all_gpus(self):
        """Decode всегда memory-bound: AI(S=0) < ridge/10 для каждой модели и GPU."""
        for name, a in self.models.items():
            ai = im.ai_decode(a)
            for gpu in im.REFERENCE_GPUS.values():
                self.assertLess(ai, gpu["ridge"] / 10, f"{name} vs {gpu['bw_gbps']}")

    def test_tps_ceiling_hand_values(self):
        """Потолок TPS (BW/байт на токен) для qwen2.5-0.5b @1008 GB/s, S=0."""
        a = self.models["qwen2.5-0.5b"]
        bw = self.gpu4090["bw_gbps"]  # 1008
        tps_fp16 = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0))          # ~1408.5
        tps_int4 = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0, bits_w=4))  # ~5634.0
        self.assertTrue(1390 < tps_fp16 < 1430, f"TPS fp16 = {tps_fp16}")
        self.assertTrue(5600 < tps_int4 < 5670, f"TPS int4 = {tps_int4}")

    def test_quant_factor_and_tps_ratio(self):
        """Квантование весов масштабирует потолок TPS линейно (memory-bound)."""
        a = self.models["qwen2.5-0.5b"]
        self.assertEqual(im.quant_factor(16, 4), 4)
        bw = self.gpu4090["bw_gbps"]
        ratio = im.tps_ceiling(bw, im.bytes_read_per_token(a, 0, bits_w=4)) / \
                im.tps_ceiling(bw, im.bytes_read_per_token(a, 0, bits_w=16))
        self.assertAlmostEqual(ratio, 4.0)

    def test_kv_crossover_hand_values(self):
        """S* где чтение KV == чтение весов (fp16/fp16)."""
        s05 = im.kv_crossover_tokens(self.models["qwen2.5-0.5b"])   # ~58_240 (tied: W меньше)
        self.assertTrue(57_000 < s05 < 60_000, f"S* = {s05}")
        s4b = im.kv_crossover_tokens(self.models["qwen3-4b"])       # ~49_280 (tied: W меньше)
        self.assertTrue(48_000 < s4b < 51_000, f"S* = {s4b}")

    def test_ai_prefill_compute_bound(self):
        """Prefill длинного промпта: AI >> ridge (compute-bound), но AI < T."""
        a = self.models["qwen2.5-0.5b"]
        ai = im.ai_prefill(a, 4096)  # ~3827 FLOP/байт (tied: W меньше -> AI выше)
        self.assertGreater(ai, 30 * self.gpu4090["ridge"])
        self.assertLess(ai, 4096)

    # --- Prefill: точные FLOPs и потолок (E2/E3) ============================
    def test_prefill_flops_hand_values(self):
        """Prefill-FLOPs gemma-4: ручной пересчёт (GEMM + ядро внимания full/SWA)."""
        a = self.models["gemma-4-12b-it"]
        # S=1: позиция 0 не читает ключей -> ровно GEMM-часть
        self.assertEqual(im.prefill_flops(a, 1), 2 * im.gemm_params(a))
        s = 131_072
        gemm = s * 2 * 10_899_947_520                                   # 2_857_355_842_682_880
        full = 8 * (4 * 8192) * s * (s - 1) / 2                         # 2_251_782_633_816_064
        swa = 40 * (4 * 4096) * (1024 * 1023 / 2 + (s - 1024) * 1024)   # 87_616_997_294_080
        self.assertEqual(im.prefill_flops(a, s), gemm + full + swa)
        # точное число @S=131072 (независимый ручной пересчёт):
        self.assertEqual(im.prefill_flops(a, s), 5_196_755_473_793_024)

    def test_prefill_flops_homogeneous_hand_value(self):
        """qwen2.5-0.5b @4096: GEMM 2_931_315_179_520 + attn 721_378_344_960 (вручную)."""
        a = self.models["qwen2.5-0.5b"]
        s = 4096
        gemm = s * 2 * im.gemm_params(a)
        attn = a.n_layers * (4 * a.q_dim) * s * (s - 1) / 2
        self.assertEqual(im.prefill_flops(a, s), gemm + attn)
        self.assertEqual(im.prefill_flops(a, s), 3_652_693_524_480)

    def test_tps_prefill_ceiling_s1_equals_compute(self):
        """При S=1 потолок prefill = tflops_eff/(2·P_gemm) (чистый compute)."""
        a = self.models["qwen2.5-0.5b"]  # gemm_params = 357_826_560 -> 1e14/715_653_120 ~ 139_733
        tps = im.tps_prefill_ceiling(100.0, a, 1)
        self.assertTrue(139_000 < tps < 140_000, f"TPS prefill S=1 = {tps}")

    # --- Блок 6: мульти-GPU сплиты — расчётная модель шины (C-20) ==========
    def test_split_gpu_bytes_hand_values(self):
        """gemma-4 @131k/262k × KV {f16, q8_0}: байт VRAM на токен на ОДНОЙ карте (N=2).

        W = P_gemm·b/8 = 10_899_947_520 · 0.61776 = 6_733_551_579.96 B (Q4_K_M).
        KV f16 @131k = (131_072·8192 + 1024·163_840)·2 = 2_483_027_968; q8 — вдвое.
        KV f16 @262k = (262_144·8192 + 1024·163_840)·2 = 4_630_511_616; q8 — вдвое.
        """
        a = self.models["gemma-4-12b-it"]
        # (W + KV)/2:
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 131_072, bits_kv=16), 4_608_289_773.9776, delta=1)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 131_072, bits_kv=8), 3_987_532_781.9776, delta=1)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 262_144, bits_kv=16), 5_682_031_597.9776, delta=1)
        self.assertAlmostEqual(im.split_gpu_bytes_per_device(a, 262_144, bits_kv=8), 4_524_403_693.9776, delta=1)

    def test_ts_ceiling_double_sg(self):
        """Идентичность: TS-потолок (N карт читают параллельно) == N × SG-потолок при том же BW."""
        a = self.models["gemma-4-12b-it"]
        for s in (131_072, 262_144):
            for kv in (16, 8):
                ratio = im.tps_ceiling_ts(397.7, a, s, bits_kv=kv) / \
                        im.tps_ceiling(397.7, im.bytes_read_per_token(a, s, bits_kv=kv))
                self.assertAlmostEqual(ratio, 2.0, delta=1e-6)

    def test_ts_comm_hand_value(self):
        """Allreduce TS: L·2·d_model·e = 48·2·3840·2 = 737_280 B/токен (PCIe; вторичный член)."""
        self.assertEqual(im.ts_comm_bytes_per_token(self.models["gemma-4-12b-it"]), 737_280)

    def test_split_homogeneous_exact(self):
        """Однородная модель: qwen2.5-0.5b @1k f16 — (W + KV)/2 в точности."""
        a = self.models["qwen2.5-0.5b"]  # W = 357_826_560·2 = 715_653_120; KV = 2·24·128·1024·2 = 12_582_912
        self.assertEqual(im.split_gpu_bytes_per_device(a, 1024), (715_653_120 + 12_582_912) / 2)

    # --- Per-layer модель: гетерогенные слои gemma-4 (tensor list ground truth) ===
    def test_layer_gemm_params_i_hand_values(self):
        """Per-layer P из таблицы; индексы = реальные номера слоёв (full на 5/11/../47)."""
        a = self.models["gemma-4-12b-it"]
        # SWA-слой: q 3840·4096 + k/v 2·3840·2048 + o 4096·3840 + FFN 3·3840·15360 = 224_133_120
        self.assertEqual(im.layer_gemm_params_i(a, 0), 224_133_120)
        # global-слой: q 3840·8192 + k 3840·512 (v нет — V=K) + o 8192·3840 + FFN = 241_827_840
        self.assertEqual(im.layer_gemm_params_i(a, 5), 241_827_840)
        self.assertTrue(im.is_full_attn_layer(a, 5))
        self.assertFalse(im.is_full_attn_layer(a, 0))
        # однородная модель — формула для каждого слоя (qwen2.5-0.5b: 14_909_440)
        q = self.models["qwen2.5-0.5b"]
        for il in range(q.n_layers):
            self.assertEqual(im.layer_gemm_params_i(q, il), im.per_layer_gemm_params(q))

    def test_kv_growth_terms_hand_values(self):
        """Ростовые члены: ∂KV/∂S и константа SWA (gemma-4 f16)."""
        a = self.models["gemma-4-12b-it"]
        # линейно растут только full-слои: 8·2·512·2 B = 16_384 на позицию — ровно наклон фита E1 (×1.00)
        self.assertEqual(im.kv_slope_bytes_per_pos(a, bits_kv=16), 16_384)
        # q8 KV: вдвое меньше → 8·2·512 = 8_192 B/pos
        self.assertEqual(im.kv_slope_bytes_per_pos(a, bits_kv=8), 8_192)
        # SWA-кольцо при полном окне: 1024·163_840·2 = 335_544_320 B (константа по S)
        self.assertEqual(im.kv_swa_const_bytes(a, bits_kv=16), 335_544_320)

    def test_layer_kv_read_hand_values(self):
        """Per-layer чтение KV на токен при позиции S: full читает все S, SWA — до окна."""
        a = self.models["gemma-4-12b-it"]
        # full-слой (il=5) @S=131_072 f16: 131_072·(2·512)·2 = 268_435_456
        self.assertEqual(im.layer_kv_read_bytes_per_token(a, 5, 131_072, bits_kv=16), 268_435_456)
        # SWA-слой (il=0): min(131_072, 1024)·(2·2048)·2 = 8_388_608 — константа за окном
        self.assertEqual(im.layer_kv_read_bytes_per_token(a, 0, 131_072, bits_kv=16), 8_388_608)
        self.assertEqual(im.layer_kv_read_bytes_per_token(a, 0, 262_144, bits_kv=16), 8_388_608)

    # --- Режимы TS: идеал vs mirror-KV патч (предсказания E3 v3) ===========
    def test_ts_mirrored_hand_values(self):
        """TS с mirror-KV: байт/карта = W/2 + SWA-кольцо/2 + S·наклон-full (full не делится).

        @131k f16: 6_733_551_579.96/2 + 335_544_320/2 + 131_072·16_384 = 5_682_031_597.98;
        потолок при BW_eff=397.7 GB/s ≈ 69.99 t/s (C-18 PP @131k: 33.19).
        @262k f16: +131_072·16_384 = 7_829_515_245.98 → ≈ 50.8 t/s (C-18 PP @256k: 26.89).
        """
        a = self.models["gemma-4-12b-it"]
        s = 131_072
        expected = 10_899_947_520 * 0.61776 / 2 + 335_544_320 / 2 + s * 16_384
        self.assertAlmostEqual(im.ts_mode_bytes_mirrored_single_head_kv(a, s, bits_kv=16), expected, delta=1)
        tps = im.tps_ceiling_ts_mirrored(397.7, a, s, bits_kv=16)
        self.assertTrue(69.5 < tps < 70.5, f"TS-mirror TPS @131k f16 = {tps}")
        tps262 = im.tps_ceiling_ts_mirrored(397.7, a, 262_144, bits_kv=16)
        self.assertTrue(50.0 < tps262 < 51.5, f"TS-mirror TPS @262k f16 = {tps262}")

    def test_ts_mirrored_equals_ideal_double_context(self):
        """Идентичность: mirror при контексте S (S≥окно) == head-split при контексте 2S —
        single kv-head удваивает эффективный наклон роста байт по шине."""
        a = self.models["gemma-4-12b-it"]
        s = 131_072
        for kv in (16, 8):
            self.assertAlmostEqual(
                im.ts_mode_bytes_mirrored_single_head_kv(a, s, bits_kv=kv),
                im.split_gpu_bytes_per_device(a, 2 * s, bits_kv=kv), delta=1e-3)

    def test_pp_and_ideal_modes_equal(self):
        """PP и идеальный TS дают одинаковые байт/карту ((W+KV)/N); различается только ускорение."""
        a = self.models["gemma-4-12b-it"]
        for s in (131_072, 262_144):
            for kv in (16, 8):
                self.assertEqual(im.pp_mode_bytes_per_device(a, s, bits_kv=kv),
                                 im.ts_mode_bytes_ideal(a, s, bits_kv=kv))

    def test_ts_mirrored_homogeneous_equals_ideal(self):
        """Однородная модель без single-head слоёв: патч mirror-KV не нужен — формула == идеал."""
        q = self.models["qwen2.5-0.5b"]
        for s in (1024, 32768):
            self.assertEqual(im.ts_mode_bytes_mirrored_single_head_kv(q, s), im.ts_mode_bytes_ideal(q, s))


if __name__ == "__main__":
    unittest.main(verbosity=2)
