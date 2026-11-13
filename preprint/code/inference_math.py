"""AoI — математическое ядро «Физики инференса LLM» (блоки 1–3, 5 роадмапа).

CPU-only, stdlib. Все архитектурные параметры моделей — ground truth с диска:
заголовки safetensors локальных фикстур (каталог задаётся переменной окружения
AOI_MODELS_DIR; см. tools/parse_headers.py). Пропуск шины и ridge-точки GPU — публичные
datasheet-значения (nvidia.com), консервативные пики.

Модель идеального decode-шага (batch=1): один проход по всем активным
GEMM-матрицам весов + чтение KV-кэша до текущей позиции S.
Приближения: чтение строки embedding O(d·e) и norms/biases (<0.5% для
большинства моделей) не считаем; FLOPs softmax'а внимания O(S·d) —
вторичный член относительно матмулей (в flops_per_token не входят).

Запуск:  python inference_math.py report   — таблица AI/TPS/KV/crossover.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    """Архитектура трансформера, параметры из header'а safetensors (ground truth).

    q_dim  = n_heads * head_dim      (размер выхода q_proj / входа o_proj)
    kv_dim = n_kv_heads * head_dim   (GQA-осведомлённо: брать из shape k/v_proj)
    """
    name: str
    d_model: int
    n_layers: int
    q_dim: int
    kv_dim: int
    inter: int
    mlp_kind: str      # swiglu | geglu | relu | gelu  (swiglu/geglu -> 3 матрицы, иначе 2)
    vocab: int
    tied: bool         # lm_head делит таблицу с embed_tokens
    dtype_bytes: float # размер элемента весов в файле (BF16/F16=2.0, F32=4.0)
    ple_table_params: int = 0      # доп. embedding-таблицы (Gemma PLE), число параметров
    per_layer_extra_gemm: int = 0  # доп. GEMM-матрицы на слой (PLE gate/proj), параметры
    layer_gemm_params: tuple[int, ...] | None = None  # гетерогенные слои: params GEMM каждого слоя
    kv_elems_full: int | None = None   # Σ по full-вниманию слоёв 2·kv_dim_i (K+V элементов/токен)
    kv_elems_swa: int | None = None    # то же для SWA-слоёв; буфер ограничен swa_window токенами
    swa_window: int = 0                # окно скользящего внимания в токенах (0 — нет SWA)
    n_layers_full_attn: int | None = None  # число full-внимания слоёв (гетерогенная арх.; None → все слои full)
    q_dim_full_layers: int | None = None   # h_q·d_head full-слоёв (gemma-4: attn_q [3840x8192] -> 8192)
    q_dim_swa_layers: int | None = None    # h_q·d_head SWA-слоёв (gemma-4: attn_q [3840x4096] -> 4096)
    full_attn_layers: tuple[int, ...] | None = None  # гетерогенная арх.: индексы full-внимания слоёв (None → однородная)

    @property
    def dtype_bits(self) -> float:
        return self.dtype_bytes * 8


def mlp_params(a: Arch) -> int:
    """Параметры MLP одного слоя: 3·d·inter для GLU-семейства, иначе 2·d·inter."""
    n = 3 if a.mlp_kind in ("swiglu", "geglu") else 2
    return n * a.d_model * a.inter


def per_layer_gemm_params(a: Arch) -> int:
    """Суммарные параметры всех GEMM-матриц одного слоя (веса, без bias/norms)."""
    return (a.d_model * a.q_dim            # q_proj
            + 2 * a.d_model * a.kv_dim     # k_proj + v_proj
            + a.q_dim * a.d_model          # o_proj
            + mlp_params(a)                # MLP
            + a.per_layer_extra_gemm)


def gemm_params(a: Arch) -> int:
    """Все матрицы, читаемые за один decode-токен (batch=1): слои + lm_head GEMM.

    При tied lm_head таблица embedding делится — decode читает одну строку O(d),
    поэтому vocab·d в объёме чтения НЕ входит.
    """
    if a.layer_gemm_params is not None:
        layers = sum(a.layer_gemm_params)
    else:
        layers = a.n_layers * per_layer_gemm_params(a)
    return layers + (0 if a.tied else a.vocab * a.d_model)


def footprint_bytes(a: Arch, bits_w: float | None = None) -> float:
    """VRAM-занимание весов. Norms/biases (<0.5% для большинства моделей) не считаем."""
    b = bits_w or a.dtype_bits
    emb_tables = 1 if a.tied else 2
    if a.layer_gemm_params is not None:
        layers = sum(a.layer_gemm_params)
    else:
        layers = a.n_layers * per_layer_gemm_params(a)
    params = (layers + a.vocab * a.d_model * emb_tables + a.ple_table_params)
    return params * b / 8


def kv_cache_bytes(a: Arch, seq_len: int, bits_kv: float | None = None) -> float:
    """K+V по всем слоям на длине контекста seq_len (batch=1), байт.

    Гетерогенные слои (gemma-4): full-внимание линейно в S; SWA-слои —
    кольцевой буфер, ограниченный окном: min(S, swa_window) токенов.
    """
    e = (bits_kv or a.dtype_bits) / 8
    if a.kv_elems_full is not None:
        elems = seq_len * a.kv_elems_full
        if a.kv_elems_swa:
            elems += min(seq_len, a.swa_window) * a.kv_elems_swa
        return elems * e
    return 2 * a.n_layers * a.kv_dim * seq_len * e


def bytes_read_per_token(a: Arch, seq_len: int, bits_w=None, bits_kv=None) -> float:
    """Байт на токен decode: веса (один проход) + чтение KV до позиции seq_len."""
    b = bits_w or a.dtype_bits
    return gemm_params(a) * b / 8 + kv_cache_bytes(a, seq_len, bits_kv)


def tps_ceiling(bw_gbps: float, bytes_read: float) -> float:
    """Потолок TPS (batch=1): BW / объём чтения на токен. Memory-bound модель."""
    return bw_gbps * 1e9 / bytes_read


def flops_per_token(a: Arch) -> float:
    """Matmul-FLOPs одного decode-шага (~2·параметры GEMM)."""
    return 2.0 * gemm_params(a)


def _attn_core_flops(qdim: int, tokens: int, window: int = 0) -> float:
    """FLOPs ядра внимания (QK^T + attn@V) prefill `tokens` токенов в одном слое.

    Позиция p читает min(p, window) ключей; на позицию — 4·(h_q·d_head)·min(p,W)
    FLOP (2×дот-произведения QK^T + 2×attn@V). GQA не меняет число операций.
    """
    S = max(tokens, 1)
    if window <= 0 or S <= window:
        s = S * (S - 1) / 2.0
    else:
        s = window * (window - 1) / 2.0 + (S - window) * window
    return 4.0 * qdim * s


def prefill_flops(a: Arch, tokens: int) -> float:
    """Суммарные FLOPs полного prefill `tokens` токенов (batch=1).

    GEMM-часть S·2·P_gemm (каждый токен = проход по весам); ядро внимания
    O(S²) у full-слоёв, O(S·окно) у SWA. Softmax O(S·d) — вторичный член,
    не входит (как и в flops_per_token).
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
    """Потолок prefill TPS (compute-bound): эффективные TFLOPS / FLOPs на токен."""
    S = max(tokens, 1)
    return tflops_eff * 1e12 / (prefill_flops(a, S) / S)


def ai_decode(a: Arch, seq_len=0, bits_w=None, bits_kv=None) -> float:
    """Arithmetic Intensity decode (FLOP/байт).

    Закон: при S=0 и нативном dtype  AI == 16/dtype_bits
    (BF16/F16 -> 1.0, F32 -> 0.5) — всегда ниже ridge любой GPU.
    """
    return flops_per_token(a) / bytes_read_per_token(a, seq_len, bits_w, bits_kv)


def ai_prefill(a: Arch, tokens: int, bits_w=None) -> float:
    """AI prefill (batch=1): FLOPs растут с T, веса читаются один раз.

    Читаем за prefill: веса + запись KV на T токенов. При больших T AI ~ T·16/bits_w.
    """
    b = bits_w or a.dtype_bits
    reads = gemm_params(a) * b / 8 + kv_cache_bytes(a, tokens, bits_w)
    return tokens * flops_per_token(a) / reads


def quant_factor(bits_from: float, bits_to: float) -> float:
    """Идеальный прирост TPS от квантования весов (линеен в объёме чтения)."""
    return bits_from / bits_to


def kv_crossover_tokens(a: Arch, bits_w=None, bits_kv=None) -> float:
    """Длина контекста S*, где чтение KV == чтение весов.

    Выше S* квантование KV начинает давать больший эффект, чем квантование весов.
    """
    b = bits_w or a.dtype_bits
    if a.kv_elems_full is not None:
        slope = a.kv_elems_full * (bits_kv or b) / 8   # только линейная часть; SWA — константа
    else:
        slope = 2 * a.n_layers * a.kv_dim * (bits_kv or b) / 8
    return gemm_params(a) * b / 8 / slope


# --- Блок 6: мульти-GPU сплиты — расчётная модель шины на устройство ========

def split_gpu_bytes_per_device(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Байт VRAM-чтения на токен decode (batch=1) на ОДНОМ устройстве при сплите.

    В идеале оба режима дают каждой карте P_gemm·b/8/N весов + KV(S)/N: PP делит
    слои вместе с их KV, TS делит матрицы — в текущем llama.cpp KV-кэш тоже
    распределён по устройствам (meta-callback pattern_kv_cache →
    GGML_BACKEND_SPLIT_AXIS_0; форма K/V-тензора [n_embd_k_gqa, kv_size], т.е.
    ось 0 = пространство kv-head'ов — проверено по master 24.10.2026). Разница
    режимов временна: PP при batch=1 сериализует половины токена (выигрыша TPS
    нет — C-18), TS параллелит их с overhead'ом allreduce по PCIe. У gemma-4
    слойный сплит симметричен (24/24, по 4 full-слоя в каждой половине) ⇒ формула точна.
    """
    b = bits_w or a.dtype_bits
    return (gemm_params(a) * b / 8 + kv_cache_bytes(a, seq_len, bits_kv)) / n_devices


def tps_ceiling_ts(bw_gbps: float, a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Идеальный потолок decode TPS при tensor split (TS): N карт читают параллельно.

    Только верхняя граница — overhead allreduce/синхронизации не учтён. Для PP
    при batch=1 цепь сериализована ⇒ реальный выигрыш ~0 (C-18), формула к PP
    неприменима как оценка TPS (только как «что было бы при параллельном чтении»).
    """
    b = bits_w or a.dtype_bits
    total = gemm_params(a) * b / 8 + kv_cache_bytes(a, seq_len, bits_kv)
    return n_devices * bw_gbps * 1e9 / total


def ts_comm_bytes_per_token(a: Arch, act_dtype_bytes: float = 2.0) -> float:
    """TS (N=2): байты по PCIe на токен — allreduce частичного hidden state [d_model]
    на каждом слое (send+receive). Вторичный член относительно VRAM-чтений
    (~0.74 MB/токен против ~9 GB @131k f16 у gemma-4); у PP аналог — межкарточный
    перенос активаций на границах слоёв того же порядка, но без allreduce.
    """
    return a.n_layers * 2 * a.d_model * act_dtype_bytes


def is_full_attn_layer(a: Arch, il: int) -> bool:
    """Слой с индексом il — full-внимание? Гетерогенная арх. (gemma-4): по таблице;
    однородная: все слои full."""
    if a.full_attn_layers is None:
        return True
    return il in a.full_attn_layers


def layer_gemm_params_i(a: Arch, il: int) -> int:
    """Параметры GEMM-матриц одного слоя (гетерогенная арх. — из таблицы слоёв)."""
    if a.layer_gemm_params is not None:
        return a.layer_gemm_params[il]
    return per_layer_gemm_params(a)


def layer_weight_bytes_per_token(a: Arch, il: int, bits_w=None) -> float:
    """Байт VRAM-чтения весов слоя il на один decode-токен (один проход по матрицам)."""
    b = bits_w or a.dtype_bits
    return layer_gemm_params_i(a, il) * b / 8


def layer_flops_per_token(a: Arch, il: int) -> float:
    """Matmul-FLOPs decode-шага в слое il (~2·P_layer)."""
    return 2.0 * layer_gemm_params_i(a, il)


def _kv_elems_per_full_layer(a: Arch) -> int:
    """K+V элементов/токен на full-внимание слой (однородность внутри типа слоёв —
    верно для gemma-4: 8·1024/8 = 1024; однородная арх. → 2·kv_dim)."""
    if a.kv_elems_full is None or not a.n_layers_full_attn:
        return 2 * a.kv_dim
    return a.kv_elems_full // a.n_layers_full_attn


def _kv_elems_per_swa_layer(a: Arch) -> int:
    """K+V элементов/токен на SWA-слой (gemma-4: 163_840/40 = 4096; нет SWA → 0)."""
    if a.kv_elems_swa is None or not a.swa_window:
        return 0
    n_swa = a.n_layers - (a.n_layers_full_attn or 0)
    return a.kv_elems_swa // max(n_swa, 1)


def layer_kv_read_bytes_per_token(a: Arch, il: int, seq_len: int, bits_kv=None) -> float:
    """Байт чтения KV-кэша в слое il на один decode-токен при позиции S.

    Full-внимание слой читает все S позиций; SWA — кольцевой буфер, ограниченный
    окном: min(S, swa_window). Это и есть «рост контекста → рост байт по шине»:
    ∂/∂S у full-слоёв = 2·kv_dim_i·e (gemma-4 global: 16_384 B/pos на всю модель),
    у SWA после заполнения окна — 0.
    """
    e = (bits_kv or a.dtype_bits) / 8
    if is_full_attn_layer(a, il):
        return seq_len * _kv_elems_per_full_layer(a) * e
    w = min(seq_len, a.swa_window)
    return w * _kv_elems_per_swa_layer(a) * e


def kv_slope_bytes_per_pos(a: Arch, bits_kv=None) -> float:
    """∂KV/∂S — байт, добавляемые в чтение KV-кэша на каждой новой позиции (decode).

    Однородная арх.: 2·L·kv_dim·e. Гетерогенная (gemma-4): линейно растут ТОЛЬКО
    full-слои → kv_elems_full·e; SWA-буфер — константа после заполнения окна.
    gemma-4 f16: 8_192·2 = 16_384 B/pos — ровно наклон фита E1 (×1.00, R²=0.998).
    """
    e = (bits_kv or a.dtype_bits) / 8
    if a.kv_elems_full is not None:
        return a.kv_elems_full * e
    return 2 * a.n_layers * a.kv_dim * e


def kv_swa_const_bytes(a: Arch, bits_kv=None) -> float:
    """Константный SWA-член при S ≥ окна: кольцевой буфер на swa_window токенов.

    Не зависит от S — рост контекста за окном не увеличивает чтение SWA-кэша.
    gemma-4 f16: 1024·163_840·2 = 335_544_320 B (≈320 MiB).
    """
    e = (bits_kv or a.dtype_bits) / 8
    if a.kv_elems_swa is None:
        return 0.0
    return a.swa_window * a.kv_elems_swa * e


def pp_mode_bytes_per_device(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Layer split (PP): байт VRAM-чтения на токен на ОДНОЙ карте = (W + KV(S))/N.

    PP делит слои вместе с их KV — каждая половина держит свои веса и KV своих слоёв;
    при batch=1 цепь между половинами сериализована ⇒ выигрыша TPS нет (C-18),
    формула даёт только байт на карту, не потолок ускорения.
    """
    return split_gpu_bytes_per_device(a, seq_len, n_devices, bits_w, bits_kv)


def ts_mode_bytes_ideal(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """TS с head-split KV во ВСЕХ слоях (работает при n_kv_heads ≥ N в каждом слое):
    байт на карту = (W + KV(S))/N — та же формула, что у PP/идеала; потолок ограничен
    allreduce (ts_comm_bytes_per_token), не VRAM-чтением."""
    return split_gpu_bytes_per_device(a, seq_len, n_devices, bits_w, bits_kv)


def ts_mode_bytes_mirrored_single_head_kv(a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """TS с mirror-KV патчем слоёв, где kv-head'ов < N (gemma-4 global: 1 kv-head).

    Механизм (source gemma4.cpp + ggml-backend-meta.cpp): GEMM-матрицы q/k/o и FFN
    разрезаны по head/строкам; SWA-KV head-split (/N — 8 kv-head'ов ≥ 2 устройств,
    проходит и в текущем master); full-внимания KV-кэш MIRRORED — каждая карта читает
    ВСЕ S позиций (head-split невозможен: 1 < N), qk-norm/rms(V) по per-row проходят
    на MIRRORED-тензорах, RoPE тоже.

    bytes/карта = W/N + min(S,W)·kv_elems_swa·e/N + S·kv_elems_full·e (full — НЕ делится).
    Однородные арх. без single-head слоёв: патч не нужен → формула идеала.
    """
    if a.kv_elems_full is None or a.kv_elems_swa is None:
        return ts_mode_bytes_ideal(a, seq_len, n_devices, bits_w, bits_kv)
    b = bits_w or a.dtype_bits
    e = (bits_kv or b) / 8
    w = gemm_params(a) * b / 8 / n_devices
    kv_swa = min(seq_len, a.swa_window) * a.kv_elems_swa * e / n_devices   # head-split
    kv_full = seq_len * a.kv_elems_full * e                                # mirror — полное чтение на карте
    return w + kv_swa + kv_full


def tps_ceiling_ts_mirrored(bw_gbps: float, a: Arch, seq_len: int, n_devices: int = 2, bits_w=None, bits_kv=None) -> float:
    """Потолок decode TPS при TS с mirror-KV (патченный режим): BW / байт на карту.

    Верхняя граница — overhead allreduce/синхронизации не учтён. Предсказания E3 v3
    (BW_eff=397.7 GB/s, две RTX 5060 Ti): gemma-4 @131k f16 ≈ 70.0 t/s, @262k ≈ 50.8;
    сравнить с C-18 PP (33.19/26.89) — TS должен дать выигрыш за счёт параллельного чтения.
    """
    return bw_gbps * 1e9 / ts_mode_bytes_mirrored_single_head_kv(a, seq_len, n_devices, bits_w, bits_kv)


# ---------------------------------------------------------------------------
# Справочные таблицы. Архитектуры — ground truth из header'ов safetensors
# локальных фикстур (AOI_MODELS_DIR, tools/parse_headers.py);
# file_mb = суммарный размер shards в MiB. GPU — публичные datasheet-значения.
# ---------------------------------------------------------------------------

def _m(name, d, l, q, kv, inter, kind, vocab, tied, dt, **kw):
    return Arch(name=name, d_model=d, n_layers=l, q_dim=q, kv_dim=kv, inter=inter,
                mlp_kind=kind, vocab=vocab, tied=tied, dtype_bytes=dt, **kw)


# gemma-4: гетерогенный порядок слоёв — full-внимание на blk.5/11/../47 (step 6),
# SWA везде else; таблица layer_gemm_params индексована реальными номерами слоёв.
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
        # GPT-NeoX naming; файл содержит U8 attention.bias-буферы [1,1,2048,2048]
        # на слой (каузальная маска) — НЕ веса, при сравнении с file_mb вычитать.
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
    # Гетерогенная архитектура (ground truth из tensor list + metadata GGUF v3;
    # подтверждено замером E1 23.10.2026 — наклон фита f16 = теории ровно ×1.00):
    # 40 локальных SWA-слоёв (sliding_window=1024, kv_dim=2048; blk.0.attn_k [3840x2048])
    # + 8 глобальных full-внимания blk.5/11/../47 step 6 (kv_dim=512;
    # blk.5.attn_k [3840x512]). Линейно в S растут только глобальные слои.
    # P_layer по tensor list (tmp-src/tensors.txt, analyze_tensors.py): SWA-слой = 14
    # тензоров / GEMM 224_133_120; GLOBAL-слой = **13** тензоров — attn_v ОТСУТСТВУЕТ
    # (в global-слоях V = K, gemma4.cpp: wv ? mm : Kcur) → P_layer(global)=241_827_840.
    # FFN = gate/up/down с GELU (LLM_FFN_GELU в gemma4.cpp; структурно GLU-семейство,
    # 3 матрицы — байты/FLOPs те же). lm_head tied с token_embd — decode читает одну
    # строку таблицы. dtype_bytes = эффективные байт/параметр файла Q4_K_M (~4.94 бита).
    "gemma-4-12b-it": {
        "arch": _m("gemma-4-12b-it", 3840, 48, 4096, 2048, 15360, "geglu", 262144, True, 0.61776,
                   layer_gemm_params=_GEMMA4_LAYER_P,   # порядок = реальные индексы слоёв (full на 5/11/../47)
                   kv_elems_full=8 * 2 * 512,       # 8_192 K+V элементов/токен — глобальные слои (линейно в S)
                   kv_elems_swa=40 * 2 * 2048,     # 163_840; буфер ограничен окном 1024
                   swa_window=1024,
                   n_layers_full_attn=8,            # full-внимания: blk.5/11/../47 (step 6)
                   q_dim_full_layers=8192,         # attn_q глобальных слоёв [3840x8192]
                   q_dim_swa_layers=4096,          # attn_q локальных слоёв [3840x4096]
                   full_attn_layers=_GEMMA4_FULL),
        "file_mb": 7_381_384_864 / 2**20,           # 7045.9 MiB (GGUF Q4_K_M)
        "source": "local GGUF v3 header + tensor list (Q4_K_M)",
    },
    # Мультимодальная: в файле есть vision/audio башни — проверка file_size N/A;
    # Arch описывает только текстовый трансформер (+ PLE-таблица и per-layer PLE GEMM).
    "gemma-4-e2b": {
        "arch": _m("gemma-4-e2b", 1536, 35, 2048, 256, 6144, "geglu", 262144, True, 2.0,
                   ple_table_params=262144 * 8960, per_layer_extra_gemm=2 * 1536 * 256),
        "file_mb": None, "source": "local safetensors header (BF16, text-only arch)",
        "text_only": True,
    },
}

REFERENCE_GPUS: dict[str, dict] = {
    # bw_gbps — пиковая пропускная способность HBM/GDDR; ridge = FLOP/байт
    # (пиковые dense TFLOPS / BW), консервативно по публичным datasheet'ам.
    "RTX-4090": {"bw_gbps": 1008.0, "ridge": 82.0},   # GDDR6X ~1 TB/s; FP32 82.6 TFLOPS
    "A100-80G": {"bw_gbps": 2039.0, "ridge": 153.0},  # HBM2e 2.039 TB/s; FP16 dense 312 TF
    "H100-SXM": {"bw_gbps": 3350.0, "ridge": 295.0},  # HBM3 3.35 TB/s; BF16 dense ~989 TF
}


def _report() -> None:
    """Таблица AI/TPS/KV/crossover по всем моделям — для посева CONCLUSIONS.md.

    «TPS nat» и KV-столбец — в нативном dtype файла (для BF16/F16 моделей = fp16;
    smollm F32 и gemma-4 Q4_K_M — эффективные биты файла, не fp16).
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
