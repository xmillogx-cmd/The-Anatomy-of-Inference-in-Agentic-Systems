@echo off
setlocal enabledelayedexpansion
REM E1: KV-cache size x quantization -> decode TPS correlation (AoI roadmap blocks 3+7)
REM Model: gemma-4-12B-it-Q4_K_M on ONE RTX 5060 Ti (CUDA device CUDA1, the non-display card)
REM Close LM Studio before running this file. It serves the Qwen Code chat backend.
REM Metric per cell = PURE decode TPS after an S-token prefix: -d {S} prefills the prefix OUTSIDE
REM the timed window (llama-bench.cpp b11819, lines 2447-2514); row label "tg64 @ d{S}".

REM Dynamic paths (any OS): AOI_ROOT is found by walking up from this file until README.md appears.
REM Env overrides: AOI_RUNNER = dir with llama-bench.exe; GGUF_MODEL = path to the .gguf file.
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

echo [E1] run start %DATE% %TIME% >> "%RAW%\run_log.txt"

REM ---- preflight 0: runner and model configured (no VRAM) ----
if defined AOI_RUNNER set "RUNNER=%AOI_RUNNER%" else set "RUNNER=%AOI_ROOT%runner-cuda"
if not exist "%RUNNER%\llama-bench.exe" goto :fail_cfg
if not defined GGUF_MODEL goto :fail_model
set "MODEL=%GGUF_MODEL%"
if not exist "%MODEL%" goto :fail_model
echo [E1] %DATE% %TIME% preflight 0 OK: RUNNER + GGUF_MODEL configured >> "%RAW%\run_log.txt"

REM ---- preflight A: runner binary works, CUDA backend loaded (no VRAM) ----
"%RUNNER%\llama-bench.exe" --version > "%RAW%\preflight_version.txt" 2>&1
findstr /c:"CUDA" "%RAW%\preflight_version.txt" >nul || goto :fail
echo [E1] %DATE% %TIME% preflight A OK: runner + CUDA backend >> "%RAW%\run_log.txt"

REM ---- preflight B: -dev name CUDA1 accepted (fake model path, no VRAM) ----
"%RUNNER%\llama-bench.exe" -m "%RUNNER%\__no_such_model__.gguf" -dev CUDA1 > "%RAW%\preflight_dev.txt" 2>&1
findstr /c:"invalid device" "%RAW%\preflight_dev.txt" >nul && goto :fail
echo [E1] %DATE% %TIME% preflight B OK: -dev CUDA1 accepted >> "%RAW%\run_log.txt"

REM ---- preflight C: one real cell f16 S=1024 r=1 (VRAM), row "tg64 @ d1024" must exist ----
"%RUNNER%\llama-bench.exe" -m "%MODEL%" -p 0 -n 64 -d 1024 -r 1 -dev CUDA1 -ctk f16 -ctv f16 > "%RAW%\preflight_f16_S1024.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"tg64 @ d1024" "%RAW%\preflight_f16_S1024.txt" >nul || goto :fail
echo [E1] %DATE% %TIME% preflight C OK: real cell f16 S=1024, tg64 @ d1024 row present >> "%RAW%\run_log.txt"

REM ---- main grid: 3 kv quants x 6 context sizes = 18 cells (OOM at some cells is data, run continues) ----
for %%Q in (f16 q8_0 q4_0) do (
  for %%S in (1024 2048 4096 8192 16384 32768) do (
    echo [E1] %DATE% %TIME% kv=%%Q S=%%S CMD: llama-bench -m MODEL -p 0 -n 64 -d %%S -r 3 -dev CUDA1 -ctk %%Q -ctv %%Q >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-bench.exe" -m "%MODEL%" -p 0 -n 64 -d %%S -r 3 -dev CUDA1 -ctk %%Q -ctv %%Q > "%RAW%\%%Q_S%%S.txt" 2>&1
    set EC=!errorlevel!
    echo [E1] %DATE% %TIME% kv=%%Q S=%%S exit=!EC! >> "%RAW%\run_log.txt"
  )
)

echo.
echo E1 done: raw files in %RAW% (incl. run_log.txt with per-cell commands/exit codes and preflight_*.txt).
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [E1] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail_cfg
echo [E1] %DATE% %TIME% ABORT at step "preflight 0 - runner": llama-bench.exe not found in RUNNER="%RUNNER%" >> "%RAW%\run_log.txt"
echo.
echo E1 ABORTED at step "preflight 0 - runner": llama-bench.exe not found in RUNNER="%RUNNER%".
echo Set env var AOI_RUNNER to a dir with llama-bench.exe (build from llama.cpp b11819, see README.md).
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail_model
echo [E1] %DATE% %TIME% ABORT at step "preflight 0 - model": GGUF_MODEL not set or file missing >> "%RAW%\run_log.txt"
echo.
echo E1 ABORTED at step "preflight 0 - model": GGUF_MODEL not set or file missing (gemma-4-12B-it-Q4_K_M.gguf for E1/E2).
echo Set env var GGUF_MODEL to the .gguf path.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [E1] %DATE% %TIME% PREFLIGHT FAILED - aborted before grid, see preflight_*.txt >> "%RAW%\run_log.txt"
echo.
echo E1 ABORTED at preflight. Inspect raw\preflight_*.txt and run_log.txt.
if not defined QWEN_NO_PAUSE pause
exit /b 1
