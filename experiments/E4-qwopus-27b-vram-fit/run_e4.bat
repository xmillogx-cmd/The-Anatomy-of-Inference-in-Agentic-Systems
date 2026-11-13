@echo off
setlocal enabledelayedexpansion
REM E4 — Qwopus3.8-27B-Coder-Flash-Q4_K_M (arch qwen35, hybrid: 65 блоков, full-attn каждый 4-й слой, KV только на них):
REM   влезает/не влезает на одну карту 16 GB и зачем нужен параллелизм когда VRAM не хватает. AoI roadmap A7.
REM Model: Jackrong_Qwopus3.8-27B-Coder-Flash-Q4_K_M.gguf by bartowski (17,442,401,824 B = 16.2 GiB weights); GGUF_MODEL env var points to the file.
REM runner-cuda b11819+mirror-KV: поддержка arch qwen35 подтверждена (llama-tokenize probe 2026-11-12: prompt_512 -> 511 токенов).
REM Промпты переиспользуются из E3 (токенизатор другой: ~0.998x от gemma-4; точные размеры видны в perf-строках «/ N runs»).
REM VRAM: закрыть ОБА инстанса LM Studio перед запуском (GPU0 инстанс обслуживает чат-сессию; GPU1 держит ~10.7 GiB).
REM Preflight A — CPU-only (безопасно тестировать во время чата); B-G требуют обе карты свободные.
REM SG-пробник и TS-пробник ПАДЕНИЯ НЕ ОЗНАЧАЮТ ABORT: ожидаемый исход фиксируется как данные в run_log.txt.
REM On ANY preflight failure: run_log.txt gets the failing step name + last 12 lines of its evidence file, echoed to console before pause.

REM Dynamic paths (any OS): AOI_ROOT is found by walking up from this file until README.md appears.
REM Env overrides: AOI_RUNNER = dir with llama-completion.exe; GGUF_MODEL = path to the .gguf file;
REM                 AOI_PROMPTS = dir with prompt_*.txt (default: shared E3 set in experiments/).
set "AOI_ROOT=%~dp0"
set /a _depth=0
:find_root
if exist "%AOI_ROOT%README.md" goto :root_found
if %_depth% GEQ 6 goto :fail_noroot
for %%d in ("%AOI_ROOT%..") do set "AOI_ROOT=%%~f\"
set /a _depth+=1
goto find_root
:root_found

set "RAW=%~dp0raw"
if not exist "%RAW%" mkdir "%RAW%"

echo [E4] run start %DATE% %TIME% >> "%RAW%\run_log.txt"
set "FAILSTEP=unknown step"
set "FAILEVID=%RAW%\run_log.txt"

REM ---- prompts: env override or shared E3 set in the repo ----
if defined AOI_PROMPTS set "PROMPTS=%AOI_PROMPTS%" else set "PROMPTS=%AOI_ROOT%experiments\E3-multi-gpu-tp-ts\raw\prompts"

REM ---- preflight 0: runner and model configured (no VRAM) ----
if defined AOI_RUNNER set "RUNNER=%AOI_RUNNER%" else set "RUNNER=%AOI_ROOT%runner-cuda"
set "FAILSTEP=preflight 0 - runner path (llama-completion.exe)" & set "FAILEVID=%RAW%\run_log.txt"
if not exist "%RUNNER%\llama-completion.exe" goto :fail
set "FAILSTEP=preflight 0 - model file (GGUF_MODEL)" & set "FAILEVID=%RAW%\run_log.txt"
if not defined GGUF_MODEL goto :fail
set "MODEL=%GGUF_MODEL%"
if not exist "%MODEL%" goto :fail
echo [E4] %DATE% %TIME% preflight 0 OK: RUNNER + GGUF_MODEL configured >> "%RAW%\run_log.txt"

REM ---- preflight A: full pipeline on CPU (-dev none, no VRAM); arch qwen35 support + both perf lines must appear ----
set "FAILSTEP=preflight A - CPU smoke (-dev none)" & set "FAILEVID=%RAW%\preflight_cpu_smoke.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 4 --ignore-eos -c 768 --perf -dev none -no-cnv --no-display-prompt > "%RAW%\preflight_cpu_smoke.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%RAW%\preflight_cpu_smoke.txt" >nul || goto :fail
findstr /c:"eval time" "%RAW%\preflight_cpu_smoke.txt" >nul || goto :fail
echo [E4] %DATE% %TIME% preflight A OK: arch qwen35 loads, both perf lines present >> "%RAW%\run_log.txt"

REM ---- preflight B: GPUs free check via nvidia-smi; abort if either card has >=2 GiB used (LM Studio) ----
set "FAILSTEP=preflight B - GPUs free check (nvidia-smi)" & set "FAILEVID=%RAW%\gpu_free.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "%RAW%\gpu_free.txt" 2>&1 || goto :fail
python -c "import sys;d=open(r'%RAW%\gpu_free.txt',encoding='utf-8',errors='replace').read().strip();rows=[l.split(',') for l in d.splitlines() if ',' in l];vals=[int(r[1]) for r in rows[:2]];ok=len(vals)==2 and max(vals)<2048;print('GPU used MiB:',vals);sys.exit(0 if ok else 1)" || goto :fail
echo [E4] %DATE% %TIME% preflight B OK: both GPUs free (see gpu_free.txt) >> "%RAW%\run_log.txt"

REM ---- SG probe: ONE card CUDA1, S=512 f16. EXPECTED failure (weights 16.2 GiB > 16 GB VRAM):
REM      verdict logged as data, NO abort on this step. ----
set "SGFILE=%RAW%\e4_sg_s512_f16.txt"
echo [E4] %DATE% %TIME% SG probe CMD: llama-completion -dev CUDA1 -c 768 -ctk f16 -ctv f16 -n 64 >> "%RAW%\run_log.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 64 --ignore-eos -c 768 --perf -dev CUDA1 -ctk f16 -ctv f16 -no-cnv --no-display-prompt > "%SGFILE%" 2>&1
set "SGEC=!errorlevel!"
echo [E4] %DATE% %TIME% SG probe exit=!SGEC! >> "%RAW%\run_log.txt"
if "!SGEC!"=="0" (
    echo [E4] %DATE% %TIME% SG VERDICT: FITS on one card 16 GB (unexpected - weights are 16.2 GiB) >> "%RAW%\run_log.txt"
) else (
    findstr /i /c:"not enough" /c:"out of memory" /c:"failed to load model weights" "%SGFILE%" >nul && echo [E4] %DATE% %TIME% SG VERDICT: DOES NOT FIT on one card 16 GB (expected - VRAM error in e4_sg_s512_f16.txt) >> "%RAW%\run_log.txt" || echo [E4] %DATE% %TIME% SG VERDICT: FAILED with non-VRAM error - inspect e4_sg_s512_f16.txt >> "%RAW%\run_log.txt"
)

REM ---- PP smoke both cards S=512 f16 (-sm layer); MUST succeed else abort (all later cells meaningless) ----
set "FAILSTEP=preflight C - PP smoke S=512 f16 both cards (-sm layer)" & set "FAILEVID=%RAW%\preflight_pp_smoke.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 64 --ignore-eos -c 768 --perf -dev CUDA0,CUDA1 -sm layer -ctk f16 -ctv f16 -no-cnv --no-display-prompt > "%RAW%\preflight_pp_smoke.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%RAW%\preflight_pp_smoke.txt" >nul || goto :fail
findstr /c:"eval time" "%RAW%\preflight_pp_smoke.txt" >nul || goto :fail
echo [E4] %DATE% %TIME% preflight C OK: PP smoke both cards (also the short-context tg/pp baseline) >> "%RAW%\run_log.txt"

REM ---- grid part 1: S=131072, PP (-sm layer), kv f16/q8_0; OOM here = valid «не влезает при этом контексте» data point (no abort) ----
for %%Q in (f16 q8_0) do (
    echo [E4] %DATE% %TIME% GRID mode=pp S=131072 kv=%%Q CMD: llama-completion -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_131072.txt" -n 64 --ignore-eos -c 131328 --perf -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -no-cnv --no-display-prompt > "%RAW%\e4_pp_s131072_%%Q.txt" 2>&1
    set EC=!errorlevel!
    echo [E4] %DATE% %TIME% GRID mode=pp S=131072 kv=%%Q exit=!EC! >> "%RAW%\run_log.txt"
)

REM ---- grid part 2: S=262144, PP (-sm layer), kv f16/q8_0; same data-point semantics as part 1 ----
for %%Q in (f16 q8_0) do (
    echo [E4] %DATE% %TIME% GRID mode=pp S=262144 kv=%%Q CMD: llama-completion -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_262144.txt" -n 64 --ignore-eos -c 262400 --perf -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -no-cnv --no-display-prompt > "%RAW%\e4_pp_s262144_%%Q.txt" 2>&1
    set EC=!errorlevel!
    echo [E4] %DATE% %TIME% GRID mode=pp S=262144 kv=%%Q exit=!EC! >> "%RAW%\run_log.txt"
)

REM ---- TS probe: -sm tensor -ts 1,1 S=512 f16; expected crash on hybrid arch (C-17 pattern for gemma-4):
REM      verdict logged as data, NO abort. Answers «какой параллелизм работает» ----
set "TSFILE=%RAW%\e4_ts_s512_f16.txt"
echo [E4] %DATE% %TIME% TS probe CMD: llama-completion -dev CUDA0,CUDA1 -sm tensor -ts 1,1 -c 768 -n 64 >> "%RAW%\run_log.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 64 --ignore-eos -c 768 --perf -dev CUDA0,CUDA1 -sm tensor -ts 1,1 -ctk f16 -ctv f16 -no-cnv --no-display-prompt > "%TSFILE%" 2>&1
set "TSEC=!errorlevel!"
echo [E4] %DATE% %TIME% TS probe exit=!TSEC! >> "%RAW%\run_log.txt"
if "!TSEC!"=="0" (
    echo [E4] %DATE% %TIME% TS VERDICT: tensor parallel WORKS on qwen35 (unexpected for hybrid arch) - perf in e4_ts_s512_f16.txt >> "%RAW%\run_log.txt"
) else (
    findstr /i /c:"GGML_ASSERT" /c:"split" "%TSFILE%" >nul && echo [E4] %DATE% %TIME% TS VERDICT: CRASHED on hybrid arch as expected (C-17 pattern; e4_ts_s512_f16.txt) >> "%RAW%\run_log.txt" || echo [E4] %DATE% %TIME% TS VERDICT: FAILED with other error - inspect e4_ts_s512_f16.txt >> "%RAW%\run_log.txt"
)

echo.
echo E4 done: raw files in %RAW% (e4_* cells + preflight_*/gpu_free.txt + run_log.txt with verdicts and per-cell commands/exit codes).
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [E4] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [E4] %DATE% %TIME% ABORT at step "%FAILSTEP%" - evidence file: %FAILEVID% >> "%RAW%\run_log.txt"
if exist "%FAILEVID%" (for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"`) do echo "[E4]   | %%L") >> "%RAW%\run_log.txt"
echo.
echo ============================================================
echo  E4 ABORTED at step: %FAILSTEP%
echo  evidence file: %FAILEVID%
REM no inline 'else' on the next line: cmd folds a quoted external command into the if-branch and swallows 'else' (verified 2026-10-23).
echo  --- last lines of evidence ---
if exist "%FAILEVID%" powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"
echo ============================================================
if not defined QWEN_NO_PAUSE pause
exit /b 1
