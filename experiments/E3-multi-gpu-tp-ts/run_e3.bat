@echo off
setlocal enabledelayedexpansion
REM E3 v2 (PP-only): TRUE multi-GPU layer split (-sm layer, pipeline parallelism); long context S=131072/262144; KV f16/q8_0 - AoI roadmap block 6.
REM Model gemma-4-12B-it-Q4_K_M on TWO RTX 5060 Ti (CUDA0+CUDA1); llama-completion one-shot runs with --perf:
REM   each cell = ONE run (-n 64 --ignore-eos) -> perf prints BOTH 'prompt eval time' (pp t/s) AND 'eval time / 63 runs' (tg t/s; b11819 -n N does N-1 decode steps).
REM v1 (llama-bench based) is archived in v1-invalid-bench/ - invalid for PP-vs-TS, see C-16. TS mode (-sm tensor -ts 1,1) crashes at graph
REM construction on gemma-4 hybrid SWA: GGML_ASSERT(src_ss[0].axis != SPLIT_AXIS_0), ggml-backend-meta.cpp:547 (by design upstream;
REM 4 identical reproductions 2026-10-23/24) -> excluded from the grid, documented as C-17.
REM VRAM: close BOTH LM Studio instances before running (GPU0 instance serves the chat session; GPU1 holds ~10.7 GiB).
REM Preflights A-B are CPU-only (safe to test during chat); C-D and the grid need both GPUs free.
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

echo [E3] run start %DATE% %TIME% >> "%RAW%\run_log.txt"
set "FAILSTEP=unknown step"
set "FAILEVID=%RAW%\run_log.txt"

REM ---- prompts: local raw\prompts if present (experiment dir), else shared E3 set in the repo ----
if defined AOI_PROMPTS set "PROMPTS=%AOI_PROMPTS%" else if exist "%RAW%\prompts\prompt_512.txt" set "PROMPTS=%RAW%\prompts" else set "PROMPTS=%AOI_ROOT%experiments\E3-multi-gpu-tp-ts\raw\prompts"

REM ---- preflight 0: runner and model configured (no VRAM) ----
if defined AOI_RUNNER set "RUNNER=%AOI_RUNNER%" else set "RUNNER=%AOI_ROOT%runner-cuda"
set "FAILSTEP=preflight 0 - runner path (llama-completion.exe)" & set "FAILEVID=%RAW%\run_log.txt"
if not exist "%RUNNER%\llama-completion.exe" goto :fail
set "FAILSTEP=preflight 0 - model file (GGUF_MODEL)" & set "FAILEVID=%RAW%\run_log.txt"
if not defined GGUF_MODEL goto :fail
set "MODEL=%GGUF_MODEL%"
if not exist "%MODEL%" goto :fail
echo [E3] %DATE% %TIME% preflight 0 OK: RUNNER + GGUF_MODEL configured >> "%RAW%\run_log.txt"

REM ---- preflight A: full pipeline on CPU (-dev none, no VRAM); both perf lines must appear ----
set "FAILSTEP=preflight A - CPU smoke (-dev none)" & set "FAILEVID=%RAW%\preflight_cpu_smoke.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 4 --ignore-eos -c 768 --perf -dev none -no-cnv --no-display-prompt > "%RAW%\preflight_cpu_smoke.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%RAW%\preflight_cpu_smoke.txt" >nul || goto :fail
findstr /c:"eval time" "%RAW%\preflight_cpu_smoke.txt" >nul || goto :fail
echo [E3] %DATE% %TIME% preflight A OK: CPU smoke, both perf lines present >> "%RAW%\run_log.txt"

REM ---- preflight B: exact token count of every prompt file (CPU-only llama-tokenize) ----
set "FAILSTEP=preflight B - prompt token-count gate" & set "FAILEVID=%RAW%\run_log.txt"
if not exist "%PROMPTS%\prompt_512.txt" goto :fail
if not exist "%PROMPTS%\prompt_1024.txt" goto :fail
if not exist "%PROMPTS%\prompt_131072.txt" goto :fail
if not exist "%PROMPTS%\prompt_262144.txt" goto :fail
set "FAILEVID=%RAW%\tokcheck_512.txt"
"%RUNNER%\llama-tokenize.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" --show-count -dev none > "%RAW%\tokcheck_512.txt" 2>&1
findstr /c:"Total number of tokens: 512" "%RAW%\tokcheck_512.txt" >nul || goto :fail
set "FAILEVID=%RAW%\tokcheck_1024.txt"
"%RUNNER%\llama-tokenize.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_1024.txt" --show-count -dev none > "%RAW%\tokcheck_1024.txt" 2>&1
findstr /c:"Total number of tokens: 1024" "%RAW%\tokcheck_1024.txt" >nul || goto :fail
set "FAILEVID=%RAW%\tokcheck_131072.txt"
"%RUNNER%\llama-tokenize.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_131072.txt" --show-count -dev none > "%RAW%\tokcheck_131072.txt" 2>&1
findstr /c:"Total number of tokens: 131072" "%RAW%\tokcheck_131072.txt" >nul || goto :fail
set "FAILEVID=%RAW%\tokcheck_262144.txt"
"%RUNNER%\llama-tokenize.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_262144.txt" --show-count -dev none > "%RAW%\tokcheck_262144.txt" 2>&1
findstr /c:"Total number of tokens: 262144" "%RAW%\tokcheck_262144.txt" >nul || goto :fail
echo [E3] %DATE% %TIME% preflight B OK: token counts exact for all four prompt files >> "%RAW%\run_log.txt"

REM ---- preflight C: PP smoke S=512 f16 both cards (-sm layer) ----
set "FAILSTEP=preflight C - PP smoke S=512 f16 both cards (-sm layer)" & set "FAILEVID=%RAW%\preflight_pp_smoke.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_512.txt" -n 64 -c 768 --perf -dev CUDA0,CUDA1 -sm layer -ctk f16 -ctv f16 -no-cnv --ignore-eos --no-display-prompt > "%RAW%\preflight_pp_smoke.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%RAW%\preflight_pp_smoke.txt" >nul || goto :fail
findstr /c:"eval time" "%RAW%\preflight_pp_smoke.txt" >nul || goto :fail
echo [E3] %DATE% %TIME% preflight C OK: PP smoke both cards >> "%RAW%\run_log.txt"

REM ---- preflight D: CLI-vs-bench equivalence gate, ONE GPU (CUDA1), S=1024 f16 n=64;
REM      tg must be 40..56 t/s with exactly 63 runs (b11819 one-shot -n N performs N-1 decode steps; E1 baseline ~47-48 @d1k) else abort before grid ----
set "FAILSTEP=preflight D - CLI-vs-bench equivalence gate (CUDA1, S=1024 f16)" & set "FAILEVID=%RAW%\preflight_equiv_s1024.txt"
"%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_1024.txt" -n 64 -c 1280 --perf -dev CUDA1 -ctk f16 -ctv f16 -no-cnv --ignore-eos --no-display-prompt > "%RAW%\preflight_equiv_s1024.txt" 2>&1
if errorlevel 1 goto :fail
findstr /c:"prompt eval time" "%RAW%\preflight_equiv_s1024.txt" >nul || goto :fail
python -c "import re,sys;t=open(r'%RAW%\preflight_equiv_s1024.txt',encoding='utf-8',errors='replace').read();m=re.search(r'eval time\s*=\s*[\d.]+ ms /\s*(\d+) runs.*?\(\s*.*?([\d.]+) tokens per second\)',t);ok=m is not None and m.group(1)=='63' and 40<=float(m.group(2))<=56;print('EQUIV tg t/s =',m.group(2) if m else 'NOMATCH');sys.exit(0 if ok else 1)" || goto :fail
echo [E3] %DATE% %TIME% preflight D OK: equivalence gate passed (see preflight_equiv_s1024.txt) >> "%RAW%\run_log.txt"

REM ---- main grid part 1: S=131072, r=2; PP (-sm layer); kv f16/q8_0; one run per cell gives pp+tg ----
for %%Q in (f16 q8_0) do (
    for /L %%R in (1,1,2) do (
      echo [E3] %DATE% %TIME% GRID mode=pp S=131072 kv=%%Q r=%%R CMD: llama-completion -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
      "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_131072.txt" -n 64 -c 131328 --perf -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -no-cnv --ignore-eos --no-display-prompt > "%RAW%\e3_pp_s131072_%%Q_r%%R.txt" 2>&1
      set EC=!errorlevel!
      echo [E3] %DATE% %TIME% GRID mode=pp S=131072 kv=%%Q r=%%R exit=!EC! >> "%RAW%\run_log.txt"
    )
)

REM ---- main grid part 2: S=262144, r=1; PP (-sm layer); kv f16/q8_0 ----
for %%Q in (f16 q8_0) do (
    echo [E3] %DATE% %TIME% GRID mode=pp S=262144 kv=%%Q r=1 CMD: llama-completion -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -n 64 --ignore-eos >> "%RAW%\run_log.txt"
    "%RUNNER%\llama-completion.exe" -m "%MODEL%" -f "%PROMPTS%\prompt_262144.txt" -n 64 -c 262400 --perf -dev CUDA0,CUDA1 -sm layer -ctk %%Q -ctv %%Q -no-cnv --ignore-eos --no-display-prompt > "%RAW%\e3_pp_s262144_%%Q_r1.txt" 2>&1
    set EC=!errorlevel!
    echo [E3] %DATE% %TIME% GRID mode=pp S=262144 kv=%%Q r=1 exit=!EC! >> "%RAW%\run_log.txt"
)

echo.
echo E3 v2 done: raw files in %RAW% (e3_* cells + preflight_*/tokcheck_* + run_log.txt with per-cell commands/exit codes).
if not defined QWEN_NO_PAUSE pause
exit /b 0

:fail_noroot
echo [E3] ABORTED at step "repo root lookup": no README.md within 6 levels above this launcher.
if not defined QWEN_NO_PAUSE pause
exit /b 1

:fail
echo [E3] %DATE% %TIME% ABORT at step "%FAILSTEP%" - evidence file: %FAILEVID% >> "%RAW%\run_log.txt"
if exist "%FAILEVID%" (for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"`) do echo "[E3]   | %%L") >> "%RAW%\run_log.txt"
echo.
echo ============================================================
echo  E3 v2 ABORTED at step: %FAILSTEP%
echo  evidence file: %FAILEVID%
REM no inline 'else' on the next line: cmd folds a quoted external command into the if-branch and swallows 'else' (verified 2026-10-23).
echo  --- last lines of evidence ---
if exist "%FAILEVID%" powershell -NoProfile -Command "Get-Content '%FAILEVID%' -Tail 12"
echo ============================================================
if not defined QWEN_NO_PAUSE pause
exit /b 1
