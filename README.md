# AoI — Anatomy of Inference in Agentic Systems

Experiment code and data for E1–E4: KV cache, quantization, prefill/decode on long context, multi-GPU tensor splitting (2× RTX 5060 Ti). The article is published separately; this repo holds the measurement pipeline behind it.

## Article — [nonndazone.ru](https://nonndazone.ru)

Published in four languages for your convenience:

- **RU** — https://nonndazone.ru/article/anatomiya-inferensa-v-agentskih-sistemah
- **EN** — https://nonndazone.ru/article/the-anatomy-of-inference-in-agentic-systems
- **ZH** — https://nonndazone.ru/article/%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E6%8E%A8%E7%90%86%E8%A7%A3%E5%89%92
- **DE** — https://nonndazone.ru/article/anatomie-des-inferencing-in-agentischen-systemen

Repository: [xmillogx-cmd/The-Anatomy-of-Inference-in-Agentic-Systems](https://github.com/xmillogx-cmd/The-Anatomy-of-Inference-in-Agentic-Systems)

## Layout

| Path | Purpose |
|---|---|
| `inference_math.py` | Math core: architecture parameters, footprint, KV growth, TPS ceiling, roofline, quantization; CLI `python inference_math.py report` |
| `tests/test_inference_math.py` | 25 reference tests (hand-computed), stdlib only |
| `tools/parse_headers.py` | Architectures from safetensors headers of local models (ground truth) |
| `tools/parse_gguf.py` | GGUF v1/v2/v3 parser: architecture, tensor shapes/types, KV types |
| `experiments/E1-kv-correlation/` … E4 | Experiment data: raw llama.cpp logs, prompts, launchers (see below) |
| `docs/mirror-kv-patch.md` | Mirror-KV patch for gemma-4 tensor splitting (E3 v3 / C-21) |

## Quick start

### Running experiments (Windows)

1. Build a llama.cpp b11819 runner — see [Dependencies → Runners](#runners-llama-cpp-b11819--not-in-the-repository).
2. Set environment variables and run the launcher:

```bat
set GGUF_MODEL=C:\models\gemma-4-12B-it-Q4_K_M.gguf
:: AOI_RUNNER defaults to %AOI_ROOT%runner-cuda; override only if your runner lives elsewhere:
:: set AOI_RUNNER=D:\builds\llama.cpp-b11819-cuda
experiments\E1-kv-correlation\run_e1.bat
```

Each launcher has a **preflight 0** gate (checks runner + model before touching the GPU), writes step name, command line and exit code of every run to `raw/run_log.txt`; on abort — tail lines of the evidence file.

## Dependencies

### Python tools — no third-party dependencies

**Python ≥ 3.9, standard library only, CPU-only.** No `pip install` required; [`requirements.txt`](requirements.txt) is a documented no-op.

| Tool | Purpose | Env var for data |
|---|---|---|
| `inference_math.py report` | AI/TPS/KV/crossover table per local model | — (architectures from headers) |
| `tools/parse_headers.py` | safetensors fixture architectures: size/dtypes, layer-0 tensors, embedding/lm_head | `AOI_MODELS_DIR` — directory containing models |
| `tools/parse_gguf.py [filter] [--all \| --tensors]` | GGUF v1/v2/v3 metadata and tensors (v3 layout hex-dump verified) | `AOI_GGUF_DIRS` — directories separated by `;` |

### Runners (llama.cpp b11819) — not in the repository

Prebuilt runners are excluded via `.gitignore` as build artifacts. Build: upstream **llama.cpp** [b11819](https://github.com/ggml-org/llama.cpp/releases/tag/b11819) + mirror-KV patch for `runner-patched`:

| Folder | Build | Used for |
|---|---|---|
| `runner-cuda/` | b11819, CUDA backend (NVIDIA driver + CUDA toolkit) | E1/E2/E4 — single GPU, CUDA backend |
| `runner-patched/` | b11819 + **mirror-KV patch** (~20 lines), TDM-GCC 10.3 + Vulkan SDK, commit d327bf1f — [`docs/mirror-kv-patch.md`](docs/mirror-kv-patch.md) | E3 v3 — TS (`-sm tensor -ts 1,1`) on gemma-4: without the patch upstream crashes deterministically (C-17) |
| `runner/` | auxiliary build | — |

Models are downloaded separately (GGUF); prompts live in the repo (`experiments/E3-multi-gpu-tp-ts/raw/prompts/`, shared set for all experiments).

## Launcher environment variables

All launchers (`experiments/*/run_*.bat`) contain no hardcoded paths: the repo root is found by walking up to `README.md` (marker) — works on any OS/layout.

| Variable | Purpose | Default |
|---|---|---|
| `AOI_RUNNER` | Folder with `llama-bench.exe` / `llama-completion.exe` / `llama-tokenize.exe` | `%AOI_ROOT%runner-cuda`; E3v3/preflight A/B — `%AOI_ROOT%runner-patched` |
| `GGUF_MODEL` | Full path to `.gguf` (required) | none — abort at preflight 0 |
| `AOI_PROMPTS` | Folder with `prompt_*.txt` | shared set in `experiments/E3-multi-gpu-tp-ts/raw/prompts` |
| `QWEN_NO_PAUSE` | Empty value disables the final pause (CI/scripts) | pause enabled |

## Tests

```bat
python tests\test_inference_math.py -v   :: from repo root: 25 tests, references — hand-computed
```

CPU-only, stdlib. The mirror-TS identity `mirror(S) == ideal(2S)` is pinned by a dedicated test (see [patch documentation](docs/mirror-kv-patch.md)).

## Data and notes

- Raw logs (`experiments/*/raw/*.txt`) contain absolute paths from llama.cpp output — they are measurement evidence and are kept as-is.
- E3 v1/v2 — invalid tensor-parallelism benchmark attempts; kept as artifacts (`v1-invalid-bench/`).

## License

Code and data — [MIT](LICENSE).
