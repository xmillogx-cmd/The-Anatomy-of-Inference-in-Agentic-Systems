@echo off
setlocal enabledelayedexpansion
REM E2: prefill TPS + KV-on-bus at long context S=65k/131k/262k, ONE GPU (CUDA1) - AoI roadmap blocks 1+7
REM Model: gemma-4-12B-it-Q4_K_M on ONE RTX 5060 Ti (CUDA device CUDA1, the non-display card)
REM Prefill cells: -p {S} -n 0 -> row "pp{S}" (pure prefill TPS; KV is written inside the timed window).
REM Decode cells: -d {S} prefills the S-token prefix OUTSIDE the timed window (b11819, as in E1);
REM                row label "tg64 @ d{S}".
REM VRAM: close the second LM Studio instance holding ~10.7 GiB on GPU1/CUDA1 before running,
REM       else S>=131k cells OOM. The chat-backend instance lives on GPU0 - it does not block this run.

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

echo [E2] run start %DATE% %TIME% >> "%RAW%\run_log.txt"

REM ---- preflight 0: runner and model configured (no VRAM) ----
if defined AOI_RUNNER set "RUNNER=%AOI_RUNNER%" else set "RUNNER=%AOI_ROOT%runner-cuda"
if not exist "%RUNNER%\llama-bench.exe" goto :fail_cfg
if not defined GGUF_MODEL goto :fail_model
set "MODEL=%GGUF_MODEL%"
if not exist "%MODEL%" goto :fail_model
echo [E2] %DATE% %TIME% preflight 0 OK: RUNNER + GGUF_MODEL configured >> "%RAW%\run_log.txt"

REM ---- preflight A: runner binary works, CUDA backend loaded (no VRAM) ----
"%RUNNER%\llama-bench.exe" --version > "%RAW%\preflight_version.txt" 2>&1
findstr /c:"CUDA" "%RAW%\preflight_version.txt" >nul || goto :fail
echo [E2] %DATE% %TIME% preflight A OK: runner + CUDA backend >> "%RAW%\run_log.txt"

REM ---- preflight B: -dev name CUDA1 accepted (fake model path, no VRAM) ----
"%RUNNER%\llama-bench.exe" -m "%RUNNER%\__no_such_model__.gguf" -dev CUDA1 > "%RAW%\preflight_dev.txt" 2>&1
findstr /c:"invalid device" "%RAW%\preflight_dev.txt" >nul && goto :fail
echo [E2] %DATE% %TIME% preflight B OK: -dev CUDA1 accepted >> "%RAW%\run_log.txt"

REM ---- preflight C: one real prefill cell f16 S=1024 r=1 (short VRAM), row "pp1024" must exist ----
"%RUNNER%\llama-bench.exe" -m "%MODEL%" -p 1024 -n 0 -r 1 -dev CUDA1 -ctk f16 -ctv f16 > "%RAW%\preflight_pp_S1024.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"pp1024" "%RAW%\preflight_pp_S1024.txt" >nul || goto :fail
echo [E2] %DATE% %TIME% preflight C OK: real prefill cell f16 S=1024, pp1024 row present >> "%RAW%\run_log.txt"

REM ---- main grid part 1: PREFILL cells (row "pp{S}"): kv x S; r=2 for S<=131k, r=1 at 262k ----
for %%Q in (f16 q8_0) do (
  for %%S in (65536 131072) do (
    echo [E2] %DATE% %TIME% PREFILL kv=%%Q S=%%S CMD: llama-bench -m MODEL -p %%S -n 0 -r 2 -dev CUDA1 -ctk %%Q -ctv %%Q >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-bench.exe" -m "%MODEL%" -p %%S -n 0 -r 2 -dev CUDA1 -ctk %%Q -ctv %%Q > "%RAW%\pp_%%Q_S%%S.txt" 2>&1
    set EC=!errorlevel!
    echo [E2] %DATE% %TIME% PREFILL kv=%%Q S=%%S exit=!EC! >> "%RAW%\run_log.txt"
  )
)
for %%Q in (f16 q8_0) do (
  echo [E2] %DATE% %TIME% PREFILL kv=%%Q S=262144 CMD: llama-bench -m MODEL -p 262144 -n 0 -r 1 -dev CUDA1 -ctk %%Q -ctv %%Q >> "%RAW%\run_log.txt"
  "%RUNNER%\llama-bench.exe" -m "%MODEL%" -p 262144 -n 0 -r 1 -dev CUDA1 -ctk %%Q -ctv %%Q > "%RAW%\pp_%%Q_S262144.txt" 2>&1
  set EC=!errorlevel!
  echo [E2] %DATE% %TIME% PREFILL kv=%%Q S=262144 exit=!EC! >> "%RAW%\run_log.txt"
)

REM ---- main grid part 2: DECODE cells (row "tg64 @ d{S}"): kv x S, r=2; OOM at some cells is data ----
for %%Q in (f16 q8_0) do (
  for %%S in (65536 131072 262144) do (
    echo [E2] %DATE% %TIME% DECODE kv=%%Q S=%%S CMD: llama-bench -m MODEL -p 0 -n 64 -d %%S -r 2 -dev CUDA1 -ctk %%Q -ctv %%Q >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-bench.exe" -m "%MODEL%" -p 0 -n 64 -d %%S -r 2 -dev CUDA1 -ctk %%Q -ctv %%Q > "%RAW%\tg_d%%S_%%Q.txt" 2>&1
    set EC=!errorlevel!
    echo [E2] %DATE% %TIME% DECODE kv=%%Q S=%%S exit=!EC! >> "%RAW%\run_log.txt"
  )
)

echo.
echo E2 done: raw files in %RAW% (pp_*/tg_* + run_log.txt with per-cell commands/exit codes and preflight_*.txt).
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [E2] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail_cfg
echo [E2] %DATE% %TIME% ABORT at step "preflight 0 - runner": llama-bench.exe not found in RUNNER="%RUNNER%" >> "%RAW%\run_log.txt"
echo.
echo E2 ABORTED at step "preflight 0 - runner": llama-bench.exe not found in RUNNER="%RUNNER%".
echo Set env var AOI_RUNNER to a dir with llama-bench.exe (build from llama.cpp b11819, see README.md).
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail_model
echo [E2] %DATE% %TIME% ABORT at step "preflight 0 - model": GGUF_MODEL not set or file missing >> "%RAW%\run_log.txt"
echo.
echo E2 ABORTED at step "preflight 0 - model": GGUF_MODEL not set or file missing (gemma-4-12B-it-Q4_K_M.gguf for E1/E2).
echo Set env var GGUF_MODEL to the .gguf path.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [E2] %DATE% %TIME% PREFLIGHT FAILED - aborted before grid, see preflight_*.txt >> "%RAW%\run_log.txt"
echo.
echo E2 ABORTED at preflight. Inspect raw\preflight_*.txt and run_log.txt.
if not defined QWEN_NO_PAUSE pause
exit /b 1
