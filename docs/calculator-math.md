# Матчасть llm-inference-calculator — формулы, константы и допущения core

Обновлено: 2026-11-12. Источник: git-клон `arxiv\llm-inference-calculator` (проверен пользователем «в живую» — цифры верны). Файлы: `src/core/{types,calibration,model,latency,memory,layout,hardware}.ts`. Назначение: точная матчасть расчётной модели для статьи + сверка с нашей байт-моделью (`inference_math.py`, `docs/mode-math-model.md`) и gap-finding'и.

## 1. Входные типы (что модель принимает)

| Тип | Ключевые поля | Примечание |
|---|---|---|
| `ModelSpec` | type dense/moe, paramsB, activeParamsB (moe), layers, hiddenSize, kvHeads, headDim, qHeads?, slidingWindow?, localLayers?, globalKvHeads/HeadDim?, mlaRatio?, linearAttentionLayers?, mambaRatio?, maxCtx | Для gemma-4: localLayers=40 + slidingWindow=1024 + globalKvHeads=1/globalHeadDim=512 — модель **умеет** описывать гетерогенное внимание |
| `GpuSpec` | vramGb, bwGbps (HBM), peakTflops[quant], nvlinkBwGbps? | нет NVLink → PCIe-фолбэк (§7) |
| `ParallelLayout` | tp × pp × ep × dp = N_gpu; EP \| TP, expert TP = TP/EP | validate: TP ≤ gpusPerNode (TP внутри узла!) |
| `Workload` | batchSize, inputLen, outputLen, prefillRatio? (=0.2 по умолчанию) | steady-state доля prefill-запросов |

## 2. Константы core (`types.ts`, `calibration.ts`, `hardware.ts`)

```
BYTES_PER_PARAM = { fp32:4, fp16:2, bf16:2, fp8:1, int8:1, int4:0.5, fp4:0.5 }   // идеализировано!
B_ACT = 2          // активации/сообщения allreduce всегда bf16 — независимо от квантования весов
PCIE_BW_GBPS = 32  // консервативный Gen4 x16 фолбэк intra-node без NVLink
DEFAULT_ALPHA_INTRA_MS = 0.01   // ~10 мкс (nccl-tests: делить на 1000)
DEFAULT_ALPHA_INTER_MS = 0.03   // ~30 мкс InfiniBand
GiB = 2^30; GB = 1e9
IDEAL: Calibration = { mfuPrefill:1, bwEffDecode:1, commEff*:1, *CommOverlap:1, alpha*:0 }  // чистый roofline-потолок
```

**Калибровка AoI (наша, из блок6):** `{mfuPrefill:0.35; bwEffDecode:0.55; commEmpIntra:1.0; commEmpInter:0.9; tpCommOverlap:0.0; epCommOverlap:0.5; ppCommOverlap:0.5; alphaIntraMs:0.4; alphaInterMs:0.03}` — tpCommOverlap=0 (все 96 allreduces/token на критическом пути) + alphaIntra 0.4 ms — по замеру E3 v3.

## 3. `deriveConstants(model, weightQuant, kvQuant)` → DerivedConstants

```
totalParams   = paramsB·1e9 (+ ngramParamsB)          // MoE: ВСЕ эксперты
activeParams  = (moe.activeParamsB ?? paramsB)·1e9    // активные на токен
wBytesTotal   = totalParams · BYTES_PER_PARAM[wQuant] // ← идеализированные байты/параметр!
flopsPerToken = 2 · activeParams                       // правило «2 FLOP на параметр»
qHeads        = qHeads ?? round(hiddenSize / headDim)  // фолбэк из README
qDim          = qHeads · headDim
```

## 4. KvGeometry — семантика KV-байтов (`model.ts::buildKvGeometry`)

Ключевые поля: `bytesPerToken` (unique-owner на позицию), `totalBytes(s)`, `readBytes(s)` — **«считает каждого потребителя»**: шаринг слоёв НЕ скрывает HBM-чтения, каждый слой читает свою долю. `bytesPerGpu(s,layout) = totalBytes/(tp·pp)`, `readBytesPerGpu` аналогично.

Формулы (стандартный путь, dense):
```
globalElems  = globalKvHeads · globalHeadDim        // gemma-4: 1×512 = 512
localElems   = kvHeads · headDim                    // gemma-4: 8×256 = 2048
bytesPerToken = 2·bKv·(globalLayers·globalElems + localLayers·localElems)
readBytes(s)  = 2·bKv·( globalLayers·globalElems·s + localLayers·localElems·min(s, slidingWindow) )
```

**Сверка с нашей моделью — точное совпадение для gemma-4 f16:**
- `readBytes(S)` = 2·2·(8·512·S + 40·2048·min(S,1024)) = **16,384·S + 335,544,320** при S≥окно — то же самое `kv_slope` и `c_swa`, что в `mode-math-model.md` §2.1;
- KV(131k) f16 = 16,384·131,072 + 335,544,320 = **2,483,027,968 B** — идентично нашему пину.
- Отличие по весам: у них `wBytesTotal = P_total · bW` (все параметры × идеализированные байты; int4=0.5), у нас — фактический Q4_K_M 0.61776 B/параметр и конвенция tied lm_head (P_gemm, без vocab·d) → их W-член систематически отличается от GGUF-реальности (gap #4).

## 5. Attention FLOPs: `attentionQuadFlops(model, kv, n)`

```
flops = 4 · qDim · ( globalLayers·n² + localLayers·n·min(n, window) )
// QK^T + AV; linear/SSM-слои вносят 0
```
Для gemma-4 @S=131k: full-attn O(S²)-часть = 2,252e15 FLOP (8 слоёв), SWA = 87.6e12 — совпадает с нашим prefill-расчётом (`mode-math-model.md` §5).

## 6. Модель задержек (`latency.ts`)

### 6.1 Коммуникации
```
intraBwBps = intraNodeBwGbps·1e9 · commEffIntra        // PCIe-фолбэк 32 GB/s без NVLink!
ringAllreduceMs(msg, tp) = ((2(tp−1)/tp) · msg / BW) + alphaIntra   // reduce-scatter + all-gather
exposedComm(t, overlap) = t·(1 − overlap)              // overlap=1 → полностью скрыто
```

### 6.2 Prefill (`prefillTime`) — TTFT
```
B = batchSize / dp;  nIn = inputLen
flopsMatmul = B · nIn · flopsPerToken                  // dense: вся длина × все параметры
flopsAttn   = attentionQuadFlops(·, nIn) · B           // O(S²)-часть глобального внимания
clusterFlops = (numGpus/dp) · peakFlopsOf(gpu,wQuant) · mfuPrefill
tComputeMs  = flops / clusterFlops

TP: msgBytes/слой = B·nIn·hiddenSize·B_ACT; reductionMs = L·ringAllreduce(msg,tp,BW)
    dense → tTpMs = 2 × reductionMs        // ДВЕ границы на слой: attention + MLP (== наши 96 allreduces/token!)
    MoE   → tTpMs = reductionMs, tEpMs = reductionMs
PP: ppMsgBytes = B·nIn·hiddenSize·B_ACT — байты ОДНОЙ boundary-ссылки ((pp−1) ссылок параллельны);
    bubble (pipeline fill/drain) НАВЕРНЯКА не моделируется (комментарий в коде: «too pessimistic»);
    BW выбор эвристикой: numGpus > gpusPerNode → inter, иначе intra
ttftMs = tCompute + exposed(tTp) + exposed(tEp) + exposed(tPp)
```

### 6.3 Decode (`decodeStepTime`) — TPOT
```
B = batchSize / dp
sHistory = inputLen + (outputLen + 1)/2          // ← СРЕДНЯЯ история по всей генерации, не финальная позиция!
weightsReadBytes = wBytesTotal / (tp·pp)         // dense: все веса с карты читаются каждый шаг
kvReadBytes      = readBytesPerGpu(sHistory, layout) · B
tBandwidthMs     = bytesPerStep / (bwGbps·1e9 · bwEffDecode)

TP: msgBytes = B·hiddenSize·B_ACT (=7,680 B при B=1); tTpMs = 2 × L·ringAllreduce   // те же 96/token
tpotMs = tBandwidth + exposed(tTp) + exposed(tEp)    // ⚠ НЕТ PP-члена — gap #1!
```

### 6.4 MoE и speculative decoding
```
expertCoverage(E, k, B) = 1 − (1 − k/E)^B   // жёсткий per-step минимум expert-weight чтений; не скрываемый пайплайном
sdDecodeStepTime: expectedTokensPerCycle = γ·α + 1; cycleTimeMs = γ·draftTpot + verifyTpot
```

## 7. VRAM-модель (`memory.ts::vramBreakdown`)

```
seqLen (steady-state) = inputLen + ⌈outputLen/2⌉     // среднее по continuous batching; PD-prefill режим → только inputLen
activationBytesPerSeq ≈ inputLen·hiddenSize·B_ACT (+ без FlashAttention: qHeads·n²·B_ACT — score-матрица на одном живом слое)
totalBytes = weights/GPU + (kvPerSeq + actPerSeq) · batch/dp + overhead(1 GiB по умолчанию)
capacityBytes = vramGb·1e9 · (1 − headroom)           // headroom, напр. 0.1 — CUDA ctx/фрагментация
bMax       = floor(budget / (kvPerSeq + actPerSeq))   // бюджет = capacity − weights − overhead
bMaxFullLen = то же при full length inputLen+outputLen // жёсткий потолок: любой запрос может вырасти до конца
```

## 8. Layout и hardware

- `gpuCount = tp·pp·dp`; `shardFactors`: все три делителя (weightNonExpert, weightExpert, kvAndActivation) = **tp·pp** — EP НЕ добавляет делитель весов (experts шардятся внутри TP: EP × expertTP = TP).
- `validateLayout`: tp·pp·dp == N_gpu; ep|tp; EP=1 для dense; **TP ≤ gpusPerNode**.
- `peakFlopsOf` бросает ошибку при отсутствии колонки (нет тихого mis-modeling); эквивалентности fp16≈bf16, fp8≈int8, int4≈fp4.

## 9. Сверка с нашей моделью и gap-finding'и (блок6 §6.4, `docs/block6-pp-vs-ts.html`)

Модельный грид считался скриптом `examples/block6-static-page.ts` на core калькулятора + AOI_CAL; сверка — по замерам C-13/C-18/E3 v2/v3:

| # | Gap | Числа |
|---|---|---|
| 1 | **PP decode без сериализации**: в `decodeStepTime` нет PP-члена → цепь карт не учитывается | model tg @131k f16 = **58.09** vs замер C-18 **33.19** (**+75%**); measured PP/SG CUDA = 0.87 согласован с сериализацией |
| 2 | **TS prefill**: split-GEMM sync penalty не моделируется (compute считается как единый кластер) | model @131k ≈ **1,547.7 t/s** vs замер ~489 (**×3.2 завышение**; +≈105 s на синхронизацию); SG-контроль совпадает (~800) — вину несёт именно TS-путь |
| 3 | **Mirror-KV**: их KV делится по tp·pp без проверки `kvHeads ≥ tp` (у gemma-4 global-слой kvHeads=1 → head-split невозможен, upstream крашится C-17); фактический режим — mirror: W/2 + c_swa/2 + S·16,384 | TS tg f16 = **16.26** vs замер 15.52 (**+5%**); q8_0 **+38…+42%** (dequant gap) |
| 4 | **int4-член идеализирован**: BYTES_PER_PARAM int4=0.5 vs GGUF Q4_K_M 0.61776 B/параметр + неконстантный Vulkan BW (320.9→247.4 GB/s по клеткам E3 v3) | SG tg ошибка ~**+14% @S=1K … −15% @S=262k** |
| 5 | **PP prefill — неожиданно точен**: mfu-offset и P2P-байты совпадают с замером «случайно» | PP pp: **−6% @131k, +0.4% @262k** (C-18 CUDA); SG-CUDA C-13 (~1/3 выше Vulkan-калиброванной кривой — mfu_CUDA ≈0.46 vs AOI_CAL 0.35) |
| 6 | **Gate JIT†**: холодный старт раннера не моделируется | @S=1k model warm prefill ≈ **1,354 t/s** против гейтовых замеров (JIT-нагрев) |

## 10. Что брать в статью

**Формулы (проверены и согласуются с нашей байт-моделью на точках gemma-4):**
`bytes/token(S) = W + readBytes(S)`; `readBytes(s) = 2·bKv·(G·gE·s + L·lE·min(s,window))`; `flopsPerToken = 2·P_active`; `attentionQuadFlops = 4·qDim·(G·n² + L·n·min(n,w))`; `ringAllreduceMs = 2(t−1)/t·msg/BW + α`; dense-TP = **2 allreduces/слой** (96/token у gemma-4); TPOT-формула с `sHistory` средним; VRAM steady-state seqLen.

**Допущения, которые надо честно оговорить в статье:**
1. PP decode — сериализация не моделируется (gap #1) → для batch=1 на consumer PCIe модель завышает PP-tg на ~75%.
2. TS prefill — sync penalty split-GEMM не моделируется (gap #2).
3. KV sharding предполагает `kvHeads ≥ tp` во всех слоях (gap #3; у gemma-4 global-слоя — нет → mirror-KV, наш патч).
4. Квантование весов идеализировано (0.5 B/параметр для int4) — реальность GGUF Q4_K_M = 0.61776.
5. Батч=1 и consumer PCIe — вне основной калибровки проекта (калибровка IDEAL + наши AOI_CAL по E3 v2/v3).
