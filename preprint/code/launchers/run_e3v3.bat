@echo off
setlocal enabledelayedexpansion
REM E3 v3 (TS + SG baseline): TRUE multi-GPU tensor parallelism (-sm tensor -ts 1,1) on gemma-4 hybrid SWA via mirror-KV patch; long context S=131072/262144; KV f16/q8_0.
REM Runner = runner-patched in the repo root by default (env AOI_RUNNER overrides); TDM-GCC + Vulkan build, mirror-KV patch applied - row 4.3 of c17-ts-plan-tracker.md.
REM Each cell = ONE one-shot run (-n 64 --ignore-eos --temp 0) -> perf prints BOTH 'prompt eval time' (pp t/s) AND 'eval time / 63 runs' (tg t/s; b11819 -n N does N-1 decode steps).
REM Grid: modes {sg, ts} x S {131072 r=2, 262144 r=1} x KV {f16, q8_0} = 12 runs. SG column = Vulkan baseline for BW_eff recalibration (CUDA 397.7 GB/s does not transfer).
REM Preflight gate 4.5 PASSED 2026-11-11: TS graph build + full decode without assert C-17; @S=1024 f16 SG tg 34.94 t/s, TS tg 18.00 / pp 361.71 (see preflight_equiv_s1024_*.txt).
REM VRAM: close BOTH LM Studio instances before running (GPU0 instance serves the chat session; GPU1 holds ~10.7 GiB). Total grid runs several hours; aborts are logged with step name + evidence tail.

REM Dynamic paths (any OS): AOI_ROOT is found by walking up from this file until README.md appears.
REM Env overrides: AOI_RUNNER = dir with llama-completion.exe (Vulkan build); GGUF_MODEL = path to the .gguf file;
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

echo [E3V3] run start %DATE% %TIME% >> "%RAW%\run_log.txt"
set "FAILSTEP=unknown step"
set "FAILEVID=%RAW%\run_log.txt"

REM ---- prompts: env override or shared E3 set in the repo ----
if defined AOI_PROMPTS set "PROMPTS=%AOI_PROMPTS%" else set "PROMPTS=%AOI_ROOT%experiments\E3-multi-gpu-tp-ts\raw\prompts"

REM ---- preflight 0: runner and model configured (no VRAM) ----
if defined AOI_RUNNER set "RUNNER=%AOI_RUNNER%" else set "RUNNER=%AOI_ROOT%runner-patched"
set "FAILSTEP=preflight 0 - runner path (llama-completion.exe)" & set "FAILEVID=%RAW%\run_log.txt"
if not exist "%RUNNER%\llama-completion.exe" goto :fail
set "FAILSTEP=preflight 0 - model file (GGUF_MODEL)" & set "FAILEVID=%RAW%\run_log.txt"
if not defined GGUF_MODEL goto :fail
set "MODEL=%GGUF_MODEL%"
if not exist "%MODEL%" goto :fail
echo [E3V3] %DATE% %TIME% preflight 0 OK: RUNNER + GGUF_MODEL configured >> "%RAW%\run_log.txt"

REM ---- preflight A: CPU smoke (-dev none, no VRAM); both perf lines must appear ----
set "FAILSTEP=preflight A - CPU smoke (-dev none)" & set "FAILEVID=%RAW%\e3v3_cpu_smoke.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 4 --ignore-eos -c 768 --perf -dev none -ctk f16 -ctv f16 --temp 0 -no-cnv --no-display-prompt > "%FAILEVID%" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%FAILEVID%" >nul || goto :fail
findstr /c:"eval time" "%FAILEVID%" >nul || goto :fail
echo [E3V3] %DATE% %TIME% preflight A OK: CPU smoke, both perf lines present >> "%RAW%\run_log.txt"

REM ---- preflight B: exact token count of the two grid prompt files (CPU-only llama-tokenize) ----
set "FAILSTEP=preflight B - prompt token-count gate" & set "FAILEVID=%RAW%\run_log.txt"
if not exist "%PROMPTS%\prompt_131072.txt" goto :fail
if not exist "%PROMPTS%\prompt_262144.txt" goto :fail
set "FAILEVID=%RAW%\tokcheck_131072.txt"
"%RUNNER%\llama-tokenize.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_131072.txt" --show-count -dev none > "%FAILEVID%" 2>&1
findstr /c:"Total number of tokens: 131072" "%FAILEVID%" >nul || goto :fail
set "FAILEVID=%RAW%\tokcheck_262144.txt"
"%RUNNER%\llama-tokenize.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_262144.txt" --show-count -dev none > "%FAILEVID%" 2>&1
findstr /c:"Total number of tokens: 262144" "%FAILEVID%" >nul || goto :fail
echo [E3V3] %DATE% %TIME% preflight B OK: token counts exact for both grid prompts >> "%RAW%\run_log.txt"

REM ---- grid part 1: SG column, S=131072 r=2; single GPU (VULKAN0); kv f16/q8_0 ----
for %%Q in (f16 q8_0) do (
    for /L %%R in (1,1,2) do (
      echo [E3V3] %DATE% %TIME% GRID mode=sg S=131072 kv=%%Q r=%%R CMD: llama-completion -dev VULKAN0 -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
      "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_131072.txt" -n 64 -c 131328 --perf -dev VULKAN0 -ctk %%Q -ctv %%Q --temp 0 -no-cnv --ignore-eos --no-display-prompt > "%RAW%\e3v3_sg_s131072_%%Q_r%%R.txt" 2>&1
      set EC=!errorlevel!
      echo [E3V3] %DATE% %TIME% GRID mode=sg S=131072 kv=%%Q r=%%R exit=!EC! >> "%RAW%\run_log.txt"
    )
)

REM ---- grid part 2: SG column, S=262144 r=1; single GPU (VULKAN0); kv f16/q8_0 ----
for %%Q in (f16 q8_0) do (
    echo [E3V3] %DATE% %TIME% GRID mode=sg S=262144 kv=%%Q r=1 CMD: llama-completion -dev VULKAN0 -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_262144.txt" -n 64 -c 262400 --perf -dev VULKAN0 -ctk %%Q -ctv %%Q --temp 0 -no-cnv --ignore-eos --no-display-prompt > "%RAW%\e3v3_sg_s262144_%%Q_r1.txt" 2>&1
    set EC=!errorlevel!
    echo [E3V3] %DATE% %TIME% GRID mode=sg S=262144 kv=%%Q r=1 exit=!EC! >> "%RAW%\run_log.txt"
)

REM ---- grid part 3: TS column, S=131072 r=2; TWO GPUs -sm tensor -ts 1,1; kv f16/q8_0 ----
for %%Q in (f16 q8_0) do (
    for /L %%R in (1,1,2) do (
      echo [E3V3] %DATE% %TIME% GRID mode=ts S=131072 kv=%%Q r=%%R CMD: llama-completion -dev VULKAN0,VULKAN1 -sm tensor -ts 1,1 -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
      "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_131072.txt" -n 64 -c 131328 --perf -dev VULKAN0,VULKAN1 -sm tensor -ts 1,1 -ctk %%Q -ctv %%Q --temp 0 -no-cnv --ignore-eos --no-display-prompt > "%RAW%\e3v3_ts_s131072_%%Q_r%%R.txt" 2>&1
      set EC=!errorlevel!
      echo [E3V3] %DATE% %TIME% GRID mode=ts S=131072 kv=%%Q r=%%R exit=!EC! >> "%RAW%\run_log.txt"
    )
)

REM ---- grid part 4: TS column, S=262144 r=1; TWO GPUs -sm tensor -ts 1,1; kv f16/q8_0 ----
for %%Q in (f16 q8_0) do (
    echo [E3V3] %DATE% %TIME% GRID mode=ts S=262144 kv=%%Q r=1 CMD: llama-completion -dev VULKAN0,VULKAN1 -sm tensor -ts 1,1 -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_262144.txt" -n 64 -c 262400 --perf -dev VULKAN0,VULKAN1 -sm tensor -ts 1,1 -ctk %%Q -ctv %%Q --temp 0 -no-cnv --ignore-eos --no-display-prompt > "%RAW%\e3v3_ts_s262144_%%Q_r1.txt" 2>&1
    set EC=!errorlevel!
    echo [E3V3] %DATE% %TIME% GRID mode=ts S=262144 kv=%%Q r=1 exit=!EC! >> "%RAW%\run_log.txt"
)

echo.
echo E3 v3 done: raw files in %RAW% (e3v3_* cells + preflight/tokcheck + run_log.txt with per-cell commands/exit codes).
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [E3V3] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [E3V3] %DATE% %TIME% ABORT at step "%FAILSTEP%" - evidence file: %FAILEVID% >> "%RAW%\run_log.txt"
if exist "%FAILEVID%" (for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"`) do echo "[E3V3]   | %%L") >> "%RAW%\run_log.txt"
echo.
echo ============================================================
echo  E3 v3 ABORTED at step: %FAILSTEP%
echo  evidence file: %FAILEVID%
REM no inline 'else' on the next line: cmd folds a quoted external command into the if-branch and swallows 'else' (verified 2026-10-23).
echo  --- last lines of evidence ---
if exist "%FAILEVID%" powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"
echo ============================================================
if not defined QWEN_NO_PAUSE pause
exit /b 1
