@echo off
setlocal

echo ============================================================
echo   TINT4 — 自动修复 torchao XPU 环境
echo ============================================================
echo.

:: ── 1. 从插件目录向上查找 ComfyUI 根目录 (含 main.py) ──────
set "COMFY_ROOT="
pushd "%~dp0"

:search_up
if exist "main.py" (
    set "COMFY_ROOT=%CD%"
    popd & goto :found_root
)
:: 盘符根目录 — 不再向上
if "%CD%"=="%CD:~0,3%" popd & goto :not_found
cd ..
goto :search_up

:not_found
echo [WARNING] 未找到 ComfyUI 根目录，回退到系统 Python
set "PYTHON_EXE=python"
goto :run

:found_root
echo [INFO] ComfyUI: %COMFY_ROOT%

:: ── 2. 查找 Python ──────────────────────────────────────────
set "PYTHON_EXE="
for %%e in (
    "%COMFY_ROOT%\.ext\python.exe"
    "%COMFY_ROOT%\python_embeded\python.exe"
    "%COMFY_ROOT%\python\python.exe"
    "%COMFY_ROOT%\venv\Scripts\python.exe"
    "%COMFY_ROOT%\python.exe"
) do (
    if exist "%%~e" if "%PYTHON_EXE%"=="" set "PYTHON_EXE=%%~e"
)
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

:run
echo [INFO] Python: %PYTHON_EXE%
echo.
"%PYTHON_EXE%" "%~dp0fix_torchao_xpu.py"
echo.
pause
