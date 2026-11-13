# AoI — Анатомия инференса в агентских систем

Препринт + весь код и данные экспериментов E1–E4: KV-кэш, квантование, prefill/decode на длинном контексте, multi-GPU tensor splitting (2× RTX 5060 Ti). Статья самодостаточна — читается без запуска чего бы то ни было.

- **Статья:** [`preprint/index.html`](preprint/index.html)
- **Репозиторий:** https://github.com/xmillogx-cmd/The-Anatomy-of-Inference-in-Agentic-Systems

## Структура

| Путь | Назначение |
|---|---|
| `ROADMAP.md` | Дорожная карта 7 блоков + матплан каждого (формулы, артефакты, статус) |
| `CONCLUSIONS.md` | Журнал выводов C-xx — фиксируется только после пройденного теста/замера |
| `inference_math.py` | Математическое ядро: параметры архитектур, footprint, KV, TPS-потолок, roofline, квантование; CLI `python inference_math.py report` |
| `tests/test_inference_math.py` | 25 эталонных тестов (ручной пересчёт), stdlib-only |
| `tools/parse_headers.py` | Архитектуры из header'ов safetensors локальных моделей (ground truth) |
| `tools/parse_gguf.py` | Разбор GGUF v1/v2/v3: архитектура, формы и типы тензоров, KV-типы |
| `preprint/` | Статья: `index.html`, инфографика (`images/`), код запуска (`code/`) с копией математического ядра для самодостаточности; `drafts/` — исходники в Markdown |
| `experiments/E1-kv-correlation/` … E4 | Данные экспериментов: raw-логи llama.cpp, промты, launchers (подробнее ниже) |
| `docs/` | Вспомогательные документы (`c17-ts-plan-tracker.md` и др.) |

## Быстрый старт

### Чтение статьи

Откройте [`preprint/index.html`](preprint/index.html) в браузере. Вся инфографика локальная (`images/`), внешних зависимостей нет.

### Запуск экспериментов (Windows)

1. Соберите раннер llama.cpp b11819 — см. [Зависимости → Раннеры](#раннеры-llama-cpp-b11819--не-входят-в-репозиторий).
2. Задайте переменные окружения и запустите launcher:

```bat
set GGUF_MODEL=C:\models\gemma-4-12B-it-Q4_K_M.gguf
:: AOI_RUNNER по умолчанию = %AOI_ROOT%runner-cuda; override только если раннер в другом месте:
:: set AOI_RUNNER=D:\builds\llama.cpp-b11819-cuda
preprint\code\launchers\run_e1.bat
```

Каждый launcher имеет **preflight 0** (проверка runner + модели до обращения к GPU), пишет в `raw/run_log.txt` имя шага, команду и exit code каждого прогона; при abort — последние строки файла-доказательства.

## Зависимости

### Python-инструменты — без сторонних зависимостей

**Python ≥ 3.9, только стандартная библиотека, CPU-only.** `pip install` не требуется; [`requirements.txt`](requirements.txt) — задокументированный no-op.

| Инструмент | Назначение | Переменная окружения для данных |
|---|---|---|
| `inference_math.py report` | Таблица AI/TPS/KV/crossover по локальным моделям | — (архитектуры берутся из header'ов) |
| `tools/parse_headers.py` | Архитектура safetensors-фикстур: size/dtypes, слои layer 0, embedding/lm_head | `AOI_MODELS_DIR` — каталог с моделями |
| `tools/parse_gguf.py [фильтр] [--all \| --tensors]` | Метаданные и тензоры GGUF v1/v2/v3 (v3-раскладка верифицирована hex-дампом) | `AOI_GGUF_DIRS` — список каталогов через `;` |

### Раннеры (llama.cpp b11819) — не входят в репозиторий

Собранные раннеры исключены `.gitignore` как бинарные артефакты сборки. Сборка: upstream **llama.cpp** [b11819](https://github.com/ggml-org/llama.cpp/releases/tag/b11819) + патч mirror-KV для `runner-patched`:

| Каталог | Сборка | Для чего |
|---|---|---|
| `runner-cuda/` | b11819, CUDA-бэкенд (NVIDIA-драйвер + CUDA toolkit) | E1/E2/E4 — одна карта, CUDA backend |
| `runner-patched/` | b11819 + **mirror-KV patch** (~20 строк), TDM-GCC 10.3 + Vulkan SDK, commit d327bf1f — [`preprint/code/mirror-kv-patch.md`](preprint/code/mirror-kv-patch.md) | E3 v3 — TS (`-sm tensor -ts 1,1`) на gemma-4: без патча upstream детерминированно крашится (C-17) |
| `runner/` | вспомогательная сборка | — |

Модели скачиваются отдельно (GGUF); промты в репозитории (`experiments/E3-multi-gpu-tp-ts/raw/prompts/`, общий набор всех экспериментов).

## Переменные окружения launchers

Все launchers (`preprint/code/launchers/*.bat` и копии в `experiments/*/`) не содержат жёстко закодированных путей: корень репо находится автоматически подъёмом вверх до файла `README.md` (маркер) — работает на любой ОС/раскладке.

| Переменная | Для чего | Дефолт |
|---|---|---|
| `AOI_RUNNER` | Каталог с `llama-bench.exe` / `llama-completion.exe` / `llama-tokenize.exe` | `%AOI_ROOT%runner-cuda`; E3v3/preflight A/B — `%AOI_ROOT%runner-patched` |
| `GGUF_MODEL` | Полный путь к `.gguf` (обязательно) | нет — abort на preflight 0 |
| `AOI_PROMPTS` | Каталог с `prompt_*.txt` | общий набор в `experiments/E3-multi-gpu-tp-ts/raw/prompts` |
| `QWEN_NO_PAUSE` | Пустое значение отключает финальную паузу bats (CI/скрипты) | пауза включена |

## Тесты

```bat
python tests\test_inference_math.py -v   :: из корня репо: 25 тестов, эталоны — ручной пересчёт
```

CPU-only, stdlib. Идентичность mirror-TS `mirror(S) == ideal(2S)` зафиксирована отдельным тестом (см. [документацию патча](preprint/code/mirror-kv-patch.md)).

## Данные и примечания

- Raw-логи (`experiments/*/raw/*.txt`) содержат абсолютные пути из вывода llama.cpp — это измерительное свидетельство, они сохраняются как есть.
- E3 v1/v2 — невалидные бенчмарки попыток tensor parallelism; сохранены как артефакты (`v1-invalid-bench/`).
- Журналы проекта: `ROADMAP.md` (A-xx) и `CONCLUSIONS.md` (C-xx).
