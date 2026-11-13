@echo off
setlocal enabledelayedexpansion
REM E4q4 — дополнение к эксперименту E4: недостающие ячейки грида с KV q4_0 (PP-режим).
REM В run_e4.bat был только f16/q8_0; здесь доводим тройку для сопоставления с E1 C-12.
REM Та же модель/флаги, что и в основном прогоне: Qwopus3.8-27B-Coder-Flash-Q4_K_M, -dev CUDA0,CUDA1 -sm layer, -n 64 --ignore-eos.
REM VRAM: закрыть ОБА инстанса LM Studio перед запуском (гейт preflight B).
REM OOM в ячейке = валидная точка «не влезает при этом контексте», не abort.

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

echo [E4q4] run start %DATE% %TIME% >> "%RAW%\run_log.txt"
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
echo [E4q4] %DATE% %TIME% preflight 0 OK: RUNNER + GGUF_MODEL configured >> "%RAW%\run_log.txt"

REM ---- preflight B: GPUs free check; abort if either card has >=2 GiB used (LM Studio) ----
set "FAILSTEP=preflight B - GPUs free check q4 supplement" & set "FAILEVID=%RAW%\gpu_free_q4.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "%RAW%\gpu_free_q4.txt" 2>&1 || goto :fail
python -c "import sys;d=open(r'%RAW%\gpu_free_q4.txt',encoding='utf-8',errors='replace').read().strip();rows=[l.split(',') for l in d.splitlines() if ',' in l];vals=[int(r[1]) for r in rows[:2]];ok=len(vals)==2 and max(vals)<2048;print('GPU used MiB:',vals);sys.exit(0 if ok else 1)" || goto :fail
echo [E4q4] %DATE% %TIME% preflight B OK: both GPUs free >> "%RAW%\run_log.txt"

REM ---- grid q4_0: S=131072 и S=262144, PP (-sm layer); та же семантика точек данных, что в E4 ----
for %%S in (131072 262144) do (
    set /a CPAD=%%S+256
    echo [E4q4] %DATE% %TIME% GRID mode=pp S=%%S kv=q4_0 CMD: llama-completion -dev CUDA0,CUDA1 -sm layer -ctk q4_0 -ctv q4_0 -n 64 --ignore-eos >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_%%S.txt" -n 64 --ignore-eos -c !CPAD! --perf -dev CUDA0,CUDA1 -sm layer -ctk q4_0 -ctv q4_0 -no-cnv --no-display-prompt > "%RAW%\e4_pp_s%%S_q4_0.txt" 2>&1
    set EC=!errorlevel!
    echo [E4q4] %DATE% %TIME% GRID mode=pp S=%%S kv=q4_0 exit=!EC! >> "%RAW%\run_log.txt"
)

echo.
echo E4q4 done: cells e4_pp_s*_q4_0.txt in %RAW%, log appended to run_log.txt with tag [E4q4].
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [E4q4] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [E4q4] %DATE% %TIME% ABORT at step "%FAILSTEP%" - evidence file: %FAILEVID% >> "%RAW%\run_log.txt"
if exist "%FAILEVID%" (for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"`) do echo "[E4q4]   | %%L") >> "%RAW%\run_log.txt"
echo.
echo ============================================================
echo  E4q4 ABORTED at step: %FAILSTEP%
echo  evidence file: %FAILEVID%
REM no inline 'else' on the next line: cmd folds a quoted external command into the if-branch and swallows 'else' (verified 2026-10-23).
echo  --- last lines of evidence ---
if exist "%FAILEVID%" powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"
echo ============================================================
if not defined QWEN_NO_PAUSE pause
exit /b 1
