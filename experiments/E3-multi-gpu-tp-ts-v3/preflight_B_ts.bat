@echo off
setlocal enabledelayedexpansion
REM E3 v3 preflight B - TS mode: TWO GPUs (VULKAN0+VULKAN1), -sm tensor -ts 1,1, S=1024 f16, n=64 -> 63 decode runs.
REM Run AFTER preflight_A_sg.bat completes. Kill BOTH LM Studio instances BEFORE running (VRAM rule).
REM Step 1 = CPU smoke (-dev none, no VRAM) validating exe+args; step 2 = real TS run. Log: raw\run_log.txt.

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

if defined AOI_RUNNER set "RUNNER=%AOI_RUNNER%" else set "RUNNER=%AOI_ROOT%runner-patched"
if defined AOI_PROMPTS set "PROMPTS=%AOI_PROMPTS%" else set "PROMPTS=%AOI_ROOT%experiments\E3-multi-gpu-tp-ts\raw\prompts"
set "RAW=%~dp0raw"
if not exist "%RAW%" mkdir "%RAW%"

echo [PF-B] run start %DATE% %TIME% >> "%RAW%\run_log.txt"
set "FAILSTEP=unknown step"
set "FAILEVID=%RAW%\run_log.txt"

REM ---- preflight 0: runner and model configured (no VRAM) ----
if not exist "%RUNNER%\llama-completion.exe" goto :fail_cfg
if not defined GGUF_MODEL goto :fail_model
set "MODEL=%GGUF_MODEL%"
if not exist "%MODEL%" goto :fail_model

REM ---- step 1: CPU smoke (-dev none, no VRAM) - validates exe + args before touching GPUs ----
set "FAILSTEP=preflight B step 1 - CPU smoke (-dev none)" & set "FAILEVID=%RAW%\preflight_B_cpu_smoke.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 4 --ignore-eos -c 768 --perf -dev none -ctk f16 -ctv f16 --temp 0 -no-cnv --no-display-prompt > "%FAILEVID%" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%FAILEVID%" >nul || goto :fail
findstr /c:"eval time" "%FAILEVID%" >nul || goto :fail
echo [PF-B] %DATE% %TIME% step 1 OK: CPU smoke, both perf lines present >> "%RAW%\run_log.txt"

REM ---- step 2: TS run on VULKAN0+VULKAN1, -sm tensor -ts 1,1, S=1024 f16 ----
set "FAILSTEP=preflight B step 2 - TS run (VULKAN0,VULKAN1, -sm tensor -ts 1,1, S=1024 f16)" & set "FAILEVID=%RAW%\preflight_equiv_s1024_ts.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_1024.txt" -n 64 -c 1280 --perf -dev VULKAN0,VULKAN1 -sm tensor -ts 1,1 -ctk f16 -ctv f16 --temp 0 -no-cnv --ignore-eos > "%FAILEVID%" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%FAILEVID%" >nul || goto :fail
findstr /c:"eval time" "%FAILEVID%" >nul || goto :fail
echo EXIT=0 >> "%FAILEVID%"
echo [PF-B] %DATE% %TIME% step 2 OK: TS run, both perf lines present (see preflight_equiv_s1024_ts.txt) >> "%RAW%\run_log.txt"

echo.
echo PREFLIGHT B done: raw\preflight_B_cpu_smoke.txt + raw\preflight_equiv_s1024_ts.txt (+ EXIT=0); log in raw\run_log.txt.
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [PF-B] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail_cfg
echo [PF-B] ABORTED at step "preflight 0 - runner": llama-completion.exe not found in RUNNER="%RUNNER%". Set env var AOI_RUNNER (runner-patched, see README.md).
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail_model
echo [PF-B] ABORTED at step "preflight 0 - model": GGUF_MODEL not set or file missing. Set env var GGUF_MODEL to the .gguf path (gemma-4-12B-it-Q4_K_M.gguf).
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [PF-B] %DATE% %TIME% ABORT at step "%FAILSTEP%" - evidence file: %FAILEVID% >> "%RAW%\run_log.txt"
if exist "%FAILEVID%" (for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"`) do echo "[PF-B]   | %%L") >> "%RAW%\run_log.txt"
echo.
echo ============================================================
echo  PREFLIGHT B ABORTED at step: %FAILSTEP%
echo  evidence file: %FAILEVID%
if exist "%FAILEVID%" powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"
echo ============================================================
if not defined QWEN_NO_PAUSE pause
exit /b 1
