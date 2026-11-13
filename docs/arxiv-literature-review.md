# Внешняя литература — кросс-чек выводов AoI (2026-11-11)

Файлы: `G:\uni\AoI\arxiv\*.pdf` (6 работ) + `arxiv/llm-inference-calculator/` (клон репо poczenai, MIT).
Назначение: постоянный кросс-чек наших выводов C-xx против внешней литературы — материал для блока 6 статьи.
Метод: PDF скачаны в `arxiv/`; количественные утверждения извлечены дословно; каждому — маппинг на конкретный вывод AoI и место в статье.

## 1. XQuant — arXiv 2508.10395
Tomar, Tiwari et al., UC Berkeley / FuriosaAI / ICSI / LBNL (авг 2025).
- Идея: квантовать **входные activations слоёв** X вместо KV-кэша; K/V рекомпозируются на лету из сохранённых квантованных входов → сразу **2× экономии памяти против f16 KV**; с межслойным сходством (XQuant-CL) до **~7.7–12.5×** при деградации ≤0.1 ppl; GQA через offline SVD проекций K/V в латентное пространство.
- Что даёт AoI: объясняет, почему naive KV-квантование — нетто-минус на нашем стеке (C-19): overhead dequant в горячем цикле decode против rematerialize XQuant, который меняет память на вычисления. Для блока 6 — «куда пойдёт будущее» после прагматичной q8_0 baseline, которую меряет E3 v3.

## 2. Multipath Memory Access (MMA) — arXiv 2512.16056
Tang, Zhang et al., Hunan University + Tencent (v2 май 2026).
- host↔GPU transfer latency-critical; peer-GPU relays (software-defined, через чужие PCIe-линки): пик H2D **245 GB/s = 4.62×** от single-PCIe baseline на 8×H20; насыщение при 6 relay-ах (bottleneck смещается в xGMI).
- Table 1, замеренная BW: PCIe Gen4 x16 = **24–26 GB/s** (теор. 32); **PCIe Gen5 x16 = 52–60 GB/s** (теор. 64); host DDR5-4800 12ch замеренная **350–400 GB/s**.
- TTFT на KV-fetch сокращается 1.14–2.38× с relay-ами.
- Что даёт AoI: квантитация стоимости нашего allreduce через host memory (P2P между двумя RTX 5060 Ti нет): связывает PCIe, а не DRAM; идея relay = будущая оптимизация при масштабе >2 GPU.

## 3. KV Offloading bottlenecks — arXiv 2601.19910
Meng, Lee, Wang, UPenn + Intel (MLSys'26).
- Аналитический framework: κ_crit = критическое отношение cached/prefill токенов, при котором исполнение становится memory-bound; типовые workload'ы превышают его на порядки → offloading всегда memory-bound.
- **99% latency уходит на transfers** при offload KV в CPU memory; GPU потребляет лишь **28% rated TDP**, обслуживая такие запросы.
- Что даёт AoI: подтверждение дизайна «KV остаётся в VRAM» — E3 v3 меряет q8_0 (квантованный, но in-VRAM) и не host-offload; это цитата для обоснования выбора в блоке 6.

## 4. Megatron-LM — arXiv 1909.08053
Shoeybi et al., NVIDIA (2019). Базовая работа по tensor parallelism.
- Intra-layer model parallelism с минимумом коммуникаций в нативном PyTorch; **76% scaling efficiency** на 8.3B / 512 GPU.
- Ключевой урок дизайна: «careful attention to the placement of layer normalization ... is critical» — они вынесли LayerNorm вне split-регионов, чтобы per-row нормализация работала на реплицированных тензорах.
- Что даёт AoI: ровно тот структурный принцип, за которым стоит assert C-17 и наш mirror-KV патч: per-row norm (attn_k_norm, rms) должен оставаться легальным → K/V кэш MIRRORED в слоях с n_kv_heads < N_devices; GEMM-веса остаются head-split. Для блока 6 — цитируем Megatron как design-precedent «norms outside split regions»; наш патч = конкретный инстанс этого принципа для hybrid-attention модели с 1 kv-head в global-слоях.

## 5. Characterizing Parallelism Strategies in LLM Inference — arXiv 2610.05305
Mirzaei & Mitra, Dell Technologies CTO office (окт 2026).
- Closed-form модель trade-off вычислений и коммуникаций для TP/PP: T_comm^TP = k(P_TP−1)·[α + (1/P_TP)(|M|/β)]; NCCL режими: latency-bound → переход → bandwidth-bound с ростом batch.
- Эффективный NVLink BW в decode ≈ **25% пика** (latency-limited), prefill ≈42%; **TP comm = 30–40% end-to-end decode latency** при малом batch.
- Доля пузыря PP: **ρ_bubble = (P_PP−1)/(W+P_PP−1)** → пренебрежима для long-context prefill, велика для small-batch decode.
- Что даёт AoI: независимое подтверждение C-18 — наш замеренный PP prefill ×1.63–1.65 ≈ 76–79% эффективности попадает ровно в их режим; «PP без выигрыша в decode» = доля пузыря при W=1. Их α-член обосновывает добавление latency-компоненты в нашу оценку стоимости allreduce (сейчас — чистый bandwidth: 737,280 B/токен ÷ BW).

## 6. Efficient Expert-Parallel Communication on PCIe-Connected Consumer GPUs (ThunderEP) — arXiv 2609.40093
Lee et al., Seoul National University (сент 2026). **Точно наша топология**: consumer RTX 4090/5090, только PCIe, без P2P/NVLink, весь трафик через CPU memory.
- Замеры: single GPU Gen5 = **56.5 GB/s**; два GPU одновременно через общий host bridge → по **35.5 GB/s** на каждого.
- Single-step dispatch/combine (ThunderEP) vs NCCL ring: средние speedup'ы **2.00×** (dispatch) / **1.53×** (combine); ring all-gather гоняет 2(N−1)·S байт на GPU против N·S у single-step → при N=6, S=8 МБ ≈ −40% байтов.
- Prefill до 1.66× быстрее vLLM; decode 1.12–1.26×.
- Что даёт AoI: твёрдая точка данных для нашего BW-допущения allreduce (§4/C-20): эффективный PCIe при одновременной работе двух GPU ≈ **35–40 GB/s**, а не spec. Наша оценка comm/токен = 737,280 B ÷ 35.5e9 + α(6–11 μs) ≈ **~30 μs** — ~1% времени токена даже при S=131k (время токена ≈ 14–37 мс). Вывод: overhead allreduce мал; выигрыш VRAM от mirror-KV доминирует. Инсайт ring vs single-step = будущая оптимизация при N>2.

## 7. llm-inference-calculator (pochenai, MIT) — репо
Клон в `arxiv/llm-inference-calculator` (shallow). TypeScript two-phase roofline: prefill compute-bound → TTFT; decode bandwidth-bound → TPOT = (W_read + KV_read)/BW_eff + T_comm.

### 7.1 Структурный кросс-чек с C-20 / mode-math-model
| Пункт | У них | У нас | Вердикт |
|---|---|---|---|
| Производные константы | W_bytes, kv_per_token, flops_per_token (3 шт.) | те же 3 в inference_math.py | ✅ идентичный фреймворк |
| KV-формула | Σ_l 2·b_kv·kv_heads_l·head_dim_l·min(S, cap_l) — per-layer cap на sliding window | hybrid SWA/global: 40×min(S,1024) + 8×S, K+V считаются отдельно | ✅ точное совпадение (их пример Gemma с local_layers+sliding_window — тот же приём) |
| TP comm | ring allreduce T_ar = n_ar·(2(t−1)/t·msg/BW + α), **α ≈ 6–11 μs**; dense: **2 full-TP allreduces/слой** | замерено ровно **2/слой** (wo-mm + ffn_down-mm PARTIAL → границы сабграфов) по graph_compute | ✅ независимое подтверждение наших 2/слой; α — недостающий член нашей оценки comm (добавляем в E3 v3, строка 4.6) |
| GPU data | RTX 5060 Ti: bwGbps=**448** spec | измеренный BW_eff fit **397.7 GB/s** (E1, R²=0.998) | ✅ калибровочный коэффициент ≈0.887 — согласуется с их подходом «spec → measured» |
| PP model | P2P hidden state B×N_in×h×b_act per stage; пузыри не моделируются | граница 7,680 B/токен @N=2 + замеренные пузыри C-18 ×1.63–1.65 | ⚠️ PP у них проще нашей эмпирики — в блоке 6 цитируем обе модели |

### 7.2 Найден и исправлен баг (gemma4_12b)
Их каталог имел `globalHeadDim: 256` с note «shared global K=V storage is represented by an effective global cache head dimension of 256 instead of 512» — интерпретация HF-флага `attention_k_eq_v: true` как общего буфера. **Неверно для runtime llama.cpp**:
- Их собственный pinned fixture (`tests/fixtures/recent-model-configs.json`, google/gemma-4-12B-it @43c9d01): `global_head_dim: 512, num_global_key_value_heads: 1, attention_k_eq_v: true`.
- Source llama.cpp (`src/models/gemma4.cpp` L243–L277): в global-слоях Vcur := Kcur алиасит только сырую проекцию; затем K = RoPE(learned attn_k_norm(raw)), V = weightless rms(raw) → **два отдельных cache-буфера** по 512-dim, пишутся в cache_k_lN/cache_v_lN.
- Замер E1: наклон VRAM f16 = **ровно 16,384 B/pos/token** (R²=0.998) = 8 global-слоёв × (512+512) элементов × 2 B — ground truth с железа.
Применено в клоне: `globalHeadDim: 256 → 512` (`src/data/models/24B.ts` L6) + note переписан с доказательствами; vitest **195/195 зелёных**. Их KV-формула с поправкой воспроизводит наш KV(S) @S=131,072 f16 = **2,483,027,968 B exactly** (ручной расчёт: 4×(8·512·131072 + 40·2048·1024)).
Влияние на их числа до поправки: KV global-слоёв недооценен в 2 раза → total VRAM @131k f16 = 1,409,286,144 вместо 2,483,027,968 (−52%); все производные TTFT/TPOT/fit для этой модели были сдвинуты.

## 8. Синтез: что литература говорит о наших утверждениях
| Утверждение AoI | Подтверждение |
|---|---|
| Механизм C-17 + дизайн mirror-KV патча (norms на MIRRORED) | Megatron norm-placement принцип (§4); LogP/α модель Dell делает per-op latency явной |
| C-18: PP prefill ×1.63–1.65, decode без выигрыша | Формула доли пузыря Dell ρ=(P−1)/(W+P−1) при W=1; LMSYS H20 данные в calibration archive калькулятора (PP2 efficiency 0.86–0.91 на prefill) |
| C-20 allreduce = 737,280 B/токен, overhead мал | SNU ThunderEP: concurrent PCIe ≈35.5 GB/s → ~30 μs/токен ≈ ~1% времени токена; MMA Table 1 Gen5 замер 52–60 GB/s (верхняя граница) |
| KV остаётся в VRAM (q8_0, не host offload) | UPenn: 99% latency на transfers, GPU при 28% TDP при offload |
| Hybrid SWA/global KV-модель (per-layer caps) | Универсальная формула калькулятора с cap_l — независимая реализация той же схемы; наш наклон E1 = ground truth по per-head dim (512 global) |
| Q4_K_M ≈ 0.61776 B/элемент | Прямых внешних данных нет — наш замер (E1); калькулятор считает spec-level, не квантованные байты |

## 9. Использование в блоке 6 статьи
- §«Почему KV-квантование не работает»: C-19 + XQuant как альтернативное направление (§1).
- §«TP на consumer PCIe»: SNU — совпадение топологии и замеренные BW; идея relay MMA как будущая работа (§2, §6).
- §«Правило выбора PP vs TS»: closed-form trade-off Dell рядом с нашими эмпирическими числами C-18/C-20 (§5).
- §«Баг C-17 и патч»: принцип norm-placement Megatron + цепка assert построчно (§4.3 mode-math-model.md) (§4).
- Калькулятор: независимая модель-кроссчек (их формула == наша; найден и исправлен баг gemma-4 12B) — готовая история «внешнего аудита» для статьи (§7).
