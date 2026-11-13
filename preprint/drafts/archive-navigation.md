# Навигация по архиву AoI — что где лежит и как использовать в статье

Обновлено: 2026-11-12. Назначение — единая карта данных проекта «The Anatomy of Inference» для написания собственной инженерной статьи (план A1–A4 в `ROADMAP.md`, секция «Статья»). Правило источников: **source of truth = `CONCLUSIONS.md`** (C-01…C-21); точные цифры — из `table.md`/`conclusion.md` экспериментов; формулы — из `inference_math.py` + `docs/mode-math-model.md`.

## 1. Корень проекта (`G:\uni\AoI\`)

| Файл | Что это | Как использовать в статье |
|---|---|---|
| `ROADMAP.md` | План блоков 1–7 + история экспериментов E1/E2/E3 v2/v3; секция «Статья» (A1–A4) — текущий план статьи | Структура аргументации: блоки = главы; строки экспериментов — готовые формулировки ожиданий vs фактов |
| `CONCLUSIONS.md` | Журнал выводов **C-01…C-21** с датами и доказательствами (единственный авторитетный источник цифр) | Каждый C-xx = готовый тезис статьи. Индекс ниже |
| `inference_math.py` | Ядро формул: параметры, footprint, KV (гетерогенные full/SWA), TPS-потолок, roofline, квантование, crossover S\*, prefill-FLOPs, модель шины PP/TS (`pp_mode_bytes_per_device`, `ts_mode_bytes_ideal`, `ts_mode_bytes_mirrored_single_head_kv`, `tps_ceiling_ts`) | Все расчётные числа статьи пересчитываются здесь — никаких «взятых из головы» цифр; CLI `report` |
| `tests/test_inference_math.py` | Эталонные значения (25/25 зелёных на 11.11): P_gemm, W-байты, KV@131k/262k, prefill-FLOPs pin @131k, сплиты PP/TS, идентичность `mirror(S)==ideal(2S)` | Доказательство воспроизводимости расчётов — ссылка в методологии статьи |
| `README.md` | Описание проекта | Контекст для введения |

### Индекс CONCLUSIONS.md (C-01…C-21) — точные заголовки

| # | Заголовок | Дата | В статье это |
|---|---|---|---|
| C-01 | Закон AI decode | 2026-10-22 | Фундамент: `AI(S=0)=16/dtype_bits` — почему decode memory-bound |
| C-02 | Decode всегда memory-bound | 2026-10-22 | Маржи до ridge 82×/153×/295× по GPU |
| C-03 | Точные значения KV-кэша | 2026-10-22 | Формула `KV = 2·L·kv_dim·S·e` в действии (9 моделей) |
| C-04 | Параметры из header'ов = размер файла ≤1% | 2026-10-22 | Методология ground truth (7 фикстур) |
| C-05 | Потолки TPS и линейность квантования | 2026-10-22 | `TPS ∝ bits_изнач/bits_нов`; ×4 при 16→4 bit |
| C-06 | KV-crossover S\* | 2026-10-22 | Точка, где KV догоняет веса по шине (gemma-4: ≈411.6k f16) |
| C-07 | Prefill compute-bound | 2026-10-22 | Разделение двух фаз — основа части 2 статьи |
| C-08 | U8 в pythia-2.8b = каузальный bias, не квантизация | 2026-10-22 | Ловушка интерпретации метрик (мелкая деталь методологии) |
| C-09 | Tied lm_head: decode читает одну строку, не таблицу | 2026-10-22 | Корректный подсчёт W в gemma-4 (конвенция проекта) |
| C-10 | Gemma-4: гетерогенное внимание меняет KV-математику | 2026-10-22, испр. 10-23 | 40 SWA + 8 full — ключ к наклону 16,384 B/pos |
| C-11 | Замер E1 подтверждает гибридную архитектуру Gemma-4 | 2026-10-23 | Фит f16: наклон ×1.00 от модели, R²=0.998 |
| C-12 | Квантование KV = чистый минус в decode TPS при S ≤ 32k | 2026-10-23 | q8_0 «цена» 30,322 B vs теория 8,704 (×3.48) — dequant overhead |
| C-13 | Prefill на длинном контексте compute-bound; TPS(S) падает по O(S²) глобального внимания | 2026-10-23 | E2: eff TFLOPS 44.6→39.1 с ростом S |
| C-14 | KV-квантование в prefill = стабильный overhead −1.5…−2% | 2026-10-23 | KV почти не влияет на prefill — контраст с decode |
| C-15 | Влияние KV-кэша на шину измеримо далеко до теоретического пересечения | 2026-10-23 | S\*≈412k, но эффект виден уже от ~16k |
| C-16 | llama-bench b11819 не делает истинный мульти-GPU параллелизм в bench-режиме | 2026-10-23 | Почему E3 v1 невалиден (архив `v1-invalid-bench/`) |
| C-17 | Tensor split (`-sm tensor`) детерминированно крашится при построении графа на gemma-4 | 2026-10-23 | Краш → механизм assert'а → mirror-KV патч (история блока 6) |
| C-18 | PP на двух RTX 5060 Ti: prefill ×1.63–1.65 single-GPU, decode batch=1 без выигрыша | 2026-10-24 | Сериализация PP при batch=1; tg 33.19/26.89 @131k/262k f16 |
| C-19 | KV-квантование остаётся чистым минусом при S≥131k в PP-режиме | 2026-10-24 | tg ratio q8/f16 = 0.771@131k, 0.671@262k — даже в PP |
| C-20 | Расчётная модель шины TS/PP: байты на карту одинаковы; TS на gemma-4 неизмерим в llama.cpp | 2026-10-24 | Формулы PP/TS + верификация по upstream master d327bf1f |
| C-21 | TS после mirror-KV патча: batch=1 на consumer PCIe хуже SG во всех клетках; BW_eff Vulkan = 247.6 GB/s | 2026-11-11 | E3 v3: tg TS/SG = 0.38–0.49; overhead ≈0.37–0.51 ms × 96 allreduces/token |

## 2. `docs/` — расчётные и плановые документы

| Файл | Выжимка | Как использовать в статье |
|---|---|---|
| `mode-math-model.md` | **Матмодель decode-токена gemma-4**: пер-строчная цепка ops из `gemma4.cpp`; байт по шине: `bytes/token(S) = W + Σ KV_read_l`; ∂/∂S = **16,384 B/pos f16** (только 8 full-слоёв); c_swa = **335,544,320 B** константа; per-layer таблица (SWA vs GLOBAL: P_layer, W-байт, KV-read @131k); PP-формула + PCIe-член 7,680 B/токен; TS: split-states тензоров, **96 allreduces/token**, ts_comm = 737,280 B/токен; механизм краша C-17 (V=K alias + per-row qk-norm → assert); mirror-KV патч и формула `W/2 + c_swa/2 + S×16,384`; идентичность `mirror(S)==ideal(2S)` | **Ядро части 2 статьи** («агентский контур»): всё про рост потока данных во времени здесь. Числа §5 — готовые рабочие примеры |
| `c17-ts-plan-tracker.md` | Трекер плана C-17/TS: все этапы закрыты (строка 5.4 ✅, журнал до 2026-11-12). История: краш → анализ upstream → локальная сборка TDM-GCC/Vulkan + патч mirror-KV (`src/llama-model.cpp` L542–L563) | Материал для раздела «как мы обходили ограничение раннера» — инженерный кейс, не только цифры |
| `arxiv-literature-review.md` | Обзор 7 работ: XQuant (2508.10395), MMA (2512.16056), KV Offloading (2601.19910), Megatron-LM (1909.08053), Dell TP/PP trade-off (2610.05305: T_comm^TP = k(P−1)[α+|M|/(P·β)], **TP comm = 30–40% decode latency** при малом batch), SNU ThunderEP (2609.40093: consumer PCIe без P2P ~35.5 GB/s/GPU через host memory), llm-inference-calculator | Теоретическое подкрепление §PCIe/мульти-GPU; Dell и ThunderEP — независимое подтверждение C-18/C-21 |
| `block6-pp-vs-ts.html` (25.3 KB) | Интерактивная страница блока 6: сетка замеров E3 v3 (vulCircles=18, cudaDiamonds=14), модельный грид калькулятора, **§6.4 gap-finding'и** (модель vs замер: PP decode +75% завышения, TS prefill ×3.2, mirror-KV tg +5%, SG tg −15…+14%) | Инфографика для статьи; §6.4 — готовый раздел «ограничения расчётной модели» (честность цифр) |
| `archive-navigation.md` | Этот файл | — |

## 3. `experiments/` — данные замеров (формат: ровно 5 файлов на эксперимент + raw/)

Структура каждого каталога: `config.json` (сетка + точные команды), `run_*.bat` (ручной лаунчер, префлохты с abort-диагностикой), `raw/` (сырые выходы + run_log.txt), `table.md` (**точные цифры**), `plot.png`, `conclusion.md`.

### E1-kv-correlation — KV × квантование → TPS (одна карта, CUDA b11819)
- Сетка: S ∈ {1k, 2k, 4k, 8k, 16k, 32k} × KV {f16, q8_0, q4_0}; fit `TPS_q(S) = BW_eff/(W + s_q·S + c_q)`.
- Ключевые факты: **наклон f16 совпадает с моделью ×1.00 (R²=0.998)** — C-11; q8_0/q4_0 — чистый минус при S≤32k: эффективная «цена» позиции 30,322 / 26,994 B против теории 8,704 / 4,608 (×3.48 / ×5.86) — dequant overhead ядер — C-12; f16 decode 48.08@1k → падение с ростом S.
- В статье: доказательство байтовой модели + честный «квантование KV не дарит скорость» (часть 1, §квантование).

### E2-long-context-prefill — prefill на длинном контексте (одна карта)
- Сетка: S ∈ {65k, 131k, 262k} × KV {f16, q8_0}; метрики pp{S} + tg64.
- Ключевые факты: prefill compute-bound, eff TFLOPS **44.6→41.8→39.1** (C-13); TPS(S) падает по O(S²) глобального внимания; KV-квантование в prefill = стабильный −1.5…−2% (C-14); влияние KV на шину измеримо далеко до S\*≈412k, уже от ~16k (C-15).
- В статье: раздел «prefill vs decode — два разных узла» + данные для инфографики падения pp.

### E3-multi-gpu-tp-ts (v2) — PP layer split на двух RTX 5060 Ti (CUDA b11819, one-shot `llama-completion`)
- Сетка: S ∈ {131k, 262k} × KV {f16, q8_0}, `-dev CUDA0,CUDA1 -sm layer`; reps r=2@131k / r=1@262k.
- Ключевые факты: PP prefill **×1.63–1.65** single-GPU (КПД 76–79% от идеала 2×); decode batch=1 — **без выигрыша**, реальный overhead −4…−7% f16 / ~−1% q8_0 (C-18); q8_0 KV остаётся минусом и на длинных S: tg ratio 0.771/0.671 (C-19). BW_eff CUDA = **397.7 GB/s** (калибровка по гейту).
- В статье: §«одна карта vs две» — PP работает только в prefill; сериализация decode при batch=1.
- `v1-invalid-bench/` — архив невалидного прогона llama-bench (C-16): как методология спасла от ложных цифр.

### E3-multi-gpu-tp-ts-v3 — SG vs TS после mirror-KV патча (локальная сборка TDM-GCC/Vulkan, `-sm tensor -ts 1,1`)
- Сетка: S ∈ {1k гейт, 131k, 262k} × KV {f16, q8_0} × режимы {sg, ts} = 12 прогонов; гейт эквивалентности A/B @S=1k.
- Ключевые факты: **TS batch=1 хуже SG во всех клетках decode** (tg TS/SG = 0.38–0.49); overhead ≈ 0.37–0.51 ms на каждый из 96 allreduces/token через host memory; BW-потолок mirror-KV недостижим — efficiency 23–43% (C-21); **BW_eff Vulkan = 247.6 GB/s** (×0.553 spec); TS prefill быстрее SG только @S=1k (**5.4×** — split-GEMM), на длинном 0.61–0.74×; PP-CUDA tg выше даже TS-Vulkan → **на consumer PCIe при batch=1 мульти-GPU не выигрывает ни в PP, ни в TS**.
- В статье: финальный вердикт §мульти-GPU + материал для «почему Dell/ThunderEP правы» (независимое подтверждение).

## 4. `arxiv/` — литература и расчётный калькулятор

### PDF-статьи (6 шт.)
| Файл | Работа | Зачем в статье |
|---|---|---|
| `2508.10395.pdf` | XQuant | Квантование KV/весов — актуальный SOTA-контекст §квантования |
| `2512.16056.pdf` | MMA | Методология мульти-GPU инференса |
| `2601.19910.pdf` | KV Offloading | Оффлоадинг KV в ОЗУ — альтернативный путь против роста контекста (часть 2) |
| `1909.08053.pdf` | Megatron-LM | Канон TP/PP-параллелизма — теоретическая рамка §мульти-GPU |
| `2610.05305.pdf` | Dell: TP vs PP trade-off | **T_comm^TP = k(P−1)[α+|M|/(P·β)]**; TP comm = 30–40% decode latency при малом batch — объясняет C-21 |
| `2609.40093.pdf` | SNU ThunderEP | Consumer PCIe без P2P: ~35.5 GB/s/GPU через host memory — объясняет overhead allreduces в E3 v3 |

### `llm-inference-calculator/` — git-клон расчётного калькулятора (проверен пользователем «в живую», цифры верны)
Ключевые файлы core (`src/core/`):
| Файл | Что внутри | Использование |
|---|---|---|
| `model.ts` | `deriveConstants(model, wQuant, kvQuant)` → {wBytesTotal = P·bW, flopsPerToken = 2·P_active, KvGeometry}; `buildKvGeometry`: bytesPerToken (unique-owner), readBytes(s) «считает каждого потребителя», bytesPerGpu/readBytesPerGpu (layout TP/PP); `attentionQuadFlops` = 4·qDim·(globalLayers·n² + localLayers·n·min(n,window)); MoE: nonExpertParamsB algebra | Формулы — сверка с нашей байт-моделью; **матчасть в `docs/calculator-math.md` (A2)** |
| `latency.ts` | `ringAllreduceMs(msg,tp) = 2(t−1)/t·msg/BW + alpha`; `prefillTime`: flopsMatmul = B·nIn·flopsPerToken + attentionQuadFlops; clusterFlops = (G/dp)·peak·mfuPrefill; TP: **dense → 2×reductionMs/слой**; PP P2P — один boundary link, bubble не моделируется; `decodeStepTime`: **sHistory = inputLen + (outputLen+1)/2** (среднее по генерации!), weightsReadBytes + kvReadBytes(sHistory); TPOT = tBandwidth + exposedTp + exposedEp — **PP-члена в decode нет** (gap #1) | Модельный грид блок6 страницы; gap-finding'и §6.4 |
| `memory.ts` | `vramBreakdown`: seqLen steady-state = inputLen + ⌈outputLen/2⌉; weights + (kv+act)·batch/dp + overhead (1 GiB); bMax/bMaxFullLen инверсия батча | VRAM-секция части 1: «сколько ОЗУ/VRAM реально нужно агенту» |
| `calibration.ts` / `types.ts` | AOI_CAL константы {mfuPrefill, bwEffDecode, commEff*, tp/ep/ppCommOverlap, alphaIntra/Inter}; BYTES_PER_PARAM; B_ACT = 2 | Калибровочные допущения — методология статьи |

## 5. Инструменты и раннеры

| Путь | Что это | Как использовать в статье |
|---|---|---|
| `tools/parse_headers.py` | Ground truth архитектур из header'ов safetensors (8 локальных моделей) | Методология: цифры берутся из весов, не из документации |
| `tools/parse_gguf.py` | Ground truth из GGUF v3: KV-секция + tensor list (667 тензоров gemma-4) | То же для GGUF; источник факта «u global-слоёв нет attn_v» |
| `tmp-src/tensors.txt` | Полный список 667 тензоров gemma-4 Q4_K_M | Доказательство P_layer по типам слоёв (SWA 13 vs GLOBAL 12 тензоров) |
| `tmp-src/analyze_tensors.py`, `verify_stage2.py`, `e3v3_analysis.py`, `e3v3_plot.py` | Разбор тензоров, живой пересчёт чисел модели до записи в документ, анализ E3 v3, отрисовка графиков | Ссылки на воспроизводимость расчётов и графиков |
| `runner-cuda/` | llama.cpp b11819 prebuilt **CUDA 13.4** (E1/E2/E3v2) | Версия раннера для воспроизведения замеров |
| `runner/` | Vulkan prebuilt фолбэк | — |
| `runner-patched/` | **Локальная сборка TDM-GCC/Vulkan из d327bf1f + патч mirror-KV** (`src/llama-model.cpp` L542–L563) | E3 v3; единственный исполняемый TS-раннер на gemma-4 |

## 6. Шпаргалка «ключевое число → где лежит» (для статьи)

| Число | Значение | Источник |
|---|---|---|
| W = 6,733,551,580 B/токен | Q4_K_M байты весов на decode-токен | `mode-math-model.md` §0; `inference_math.py` |
| ∂bytes/∂S = **16,384 B/pos** (f16) / 8,192 (q8) | Рост чтения на каждую новую позицию контекста | `mode-math-model.md` §2.1; fit E1 ×1.00 |
| c_swa = **335,544,320 B** | Константный SWA-член при S ≥ окно 1024 | `mode-math-model.md` §2.1 |
| KV(131k) f16 = **2,483,027,968 B**; (262k) = 4,630,511,616 | Полный KV-кэш на позиции S | `mode-math-model.md` §2.3; пин в тестах |
| bytes/token @131k f16 = **9,216,579,548 B** | W + KV | ibid. |
| S\* ≈ **411,600** | Пересечение KV=веса (вне контекста 262k) — веса доминируют в рабочем диапазоне | `mode-math-model.md` §2.3; C-06/C-15 |
| BW_eff CUDA = **397.7 GB/s**; Vulkan = **247.6 GB/s** (×0.553 spec) | Калиброванные эффективные шины RTX 5060 Ti | гейт E3v2 / гейт E3 v3 |
| PP: prefill ×1.63–1.65; decode без выигрыша (−4…−7%) | C-18 | `experiments/E3-multi-gpu-tp-ts/table.md` |
| TS tg/SG = **0.38–0.49**; overhead 0.37–0.51 ms × 96 allreduces/token | C-21 | `experiments/E3-multi-gpu-tp-ts-v3/table.md` |
| ts_comm = **737,280 B/токен** (48 слоёв × 2 × d_model 3840 × e) | Байт allreduces по PCIe в TS | `mode-math-model.md` §4.1; пин в тестах |
| Prefill FLOPs @131k = **5,196,755,473,793,024** (GEMM 2.857e15 + full-attn O(S²) 2.252e15 + SWA 87.6e12) | Вычислительная цена prefill | `mode-math-model.md` §5; пин в тестах |
| decode FLOPs/токен = **21,799,895,040**; AI @S=0 ≈ 3.2 FLOP/B (Q4_K_M) | Глубокий memory-bound | `inference_math.py`; C-01/C-02 |
| Эфф. TFLOPS prefill: **44.6→41.8→39.1** @65k/131k/262k f16 (CUDA) | C-13 | `experiments/E2-long-context-prefill/table.md` |
