# AoI — Анатомия инференса в агентских систем

Корневая папка проекта «The Anatomy of Inference». Создана: 2026-10-22.

Статья (препринт, HTML): [`preprint/index.html`](preprint/index.html) — «Анатомия инференса в агентских систем: шина памяти, квантование и параллелизм». Код и данные всех экспериментов E1–E4 идут в репозитории; препринт самодостаточен для чтения без запуска.

## Структура

| Путь | Назначение |
|---|---|
| `ROADMAP.md` | Дорожная карта 7 блоков + матплан каждого (формулы, артефакты, статус) |
| `inference_math.py` | Математическое ядро: параметры, footprint, KV, TPS-потолок, roofline, квантование; CLI `python inference_math.py report` |
| `CONCLUSIONS.md` | Журнал выводов C-xx: фиксируется только после пройденного теста/замера |
| `tools/parse_headers.py` | Ground truth архитектур из header'ов safetensors локальных моделей (CPU-only) |
| `tools/parse_gguf.py` | Разбор каталогов GGUF (архитектура, размеры слоёв, KV-типы) по env `AOI_GGUF_DIRS` |
| `tests/test_inference_math.py` | Эталонные значения (ручной пересчёт): 25 тестов, запуск `python tests\test_inference_math.py -v` из корня репо; CPU-only, stdlib |
| `preprint/` | Статья: `index.html`, инфографика (`images/`), код запуска (`code/launchers/*.bat`) + копия математического ядра и инструментов для самодостаточности препринта; `drafts/` — исходники статьи в Markdown |
| `experiments/E1-kv-correlation/` | E1 — корреляция KV-типа и скорости: raw-логи llama.cpp, промт-файлы, launcher |
| `experiments/E2-long-context-prefill/` | E2 — prefill на длинном контексте (S=512…262144) |
| `experiments/E3-multi-gpu-tp-ts/` | E3 v1/v2 — попытка tensor parallelism (невалидные бенчмарки, сохранены как артефакты; `v1-invalid-bench/`) |
| `experiments/E3-multi-gpu-tp-ts-v3/` | E3 v3 — корректный TS (`-sm tensor -ts 1,1`) через mirror-KV патч + SG baseline; preflight A/B bats |
| `experiments/E4-qwopus-27b-vram-fit/` | E4 — Qwopus3.8-27B (arch qwen35): влезает ли на одну карту 16 GB, PP vs TS, KV f16/q8_0/q4_0 |
| `docs/` | Вспомогательные документы: c17-ts-plan-tracker.md и др. |

## Запуск экспериментов (динамические пути)

Все launchers (`preprint/code/launchers/*.bat`, копии в каталогах `experiments/*/`) не содержат жёстко закодированных путей: корень репо находится автоматически подъёмом вверх до файла `README.md` (маркер). Переменные окружения переопределяют дефолты:

| Переменная | Для чего | Дефолт |
|---|---|---|
| `AOI_RUNNER` | Каталог с `llama-bench.exe` / `llama-completion.exe` / `llama-tokenize.exe` | `%AOI_ROOT%runner-cuda`; для E3v3 и preflight A/B — `%AOI_ROOT%runner-patched` |
| `GGUF_MODEL` | Полный путь к `.gguf` модели (обязательно) | нет — без неё launcher abort на preflight 0 |
| `AOI_PROMPTS` | Каталог с `prompt_*.txt` | `%AOI_ROOT%experiments\E3-multi-gpu-tp-ts\raw\prompts` |
| `QWEN_NO_PAUSE` | Пустое значение отключает `pause` в конце bats (для CI/скриптов) | пауза включена |

Python-инструменты аналогично не зависят от конкретной машины:

| Переменная | Для чего |
|---|---|
| `AOI_MODELS_DIR` | Каталог с safetensors-фикстурами (`tools/parse_headers.py`) |
| `AOI_GGUF_DIRS` | Колонноразделённый список каталогов GGUF (`tools/parse_gguf.py`) |

Пример (Windows):

```bat
set GGUF_MODEL=C:\models\bartowski\Qwen3.5-4B-GPTQ-Q4_K_M.gguf
:: AOI_RUNNER по умолчанию = %AOI_ROOT%runner-cuda; override только если раннер лежит в другом месте:
:: set AOI_RUNNER=D:\builds\llama.cpp-b11819-cuda
preprint\code\launchers\run_e1.bat
```

Каждый launcher имеет preflight 0 (проверка runner + модели до обращения к GPU) и пишет в `raw/run_log.txt` имя шага, команду и exit code каждого прогона; при abort — последние строки файла-доказательства.

## Собранные раннеры не входят в репозиторий

Каталоги `runner-cuda/`, `runner-patched/`, `runner/` (собранное llama.cpp) исключены `.gitignore` как бинарные артефакты сборки. Сборка: upstream **llama.cpp b11819** + патч mirror-KV — см. [`preprint/code/mirror-kv-patch.md`](preprint/code/mirror-kv-patch.md) (`runner-patched`, TDM-GCC, Vulkan; нужен для E3v3). `runner-cuda` — тот же b11819 с CUDA-бэкендом (E1/E2/E4).

## Примечания о данных

- Raw-логи (`experiments/*/raw/*.txt`) содержат абсолютные пути из вывода llama.cpp (`load_backend: loaded CUDA backend from ...`) — это измерительное свидетельство, они сохраняются как есть.
- `tmp-src/` (локальная рабочая копия upstream llama.cpp) и `src-llama-cpp-b11819/` исключены `.gitignore`; при необходимости скачиваются заново с GitHub releases b11819.
- Промты: `experiments/E3-multi-gpu-tp-ts/raw/prompts/prompt_{512,1024,...,262144}.txt` — общий набор всех экспериментов (переиспользуется через `AOI_PROMPTS`).
