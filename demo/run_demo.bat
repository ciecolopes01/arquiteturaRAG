@echo off
chcp 65001 > nul
echo ========================================================
echo   RAG Multimodal Enterprise - Inicializando Demo Local
echo ========================================================
echo.

cd /d "%~dp0\.."

:: Verifica se o ambiente virtual existe
if exist ".venv\Scripts\activate.bat" (
    echo [*] Ativando ambiente virtual .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [*] Usando Python do sistema...
)

echo [*] Iniciando servidor FastAPI em http://localhost:8000 ...
echo [*] Pressione CTRL+C para encerrar o servidor a qualquer momento.
echo.

:: Abre o navegador automaticamente apos 2 segundos
start "" cmd /c "timeout /t 2 /nobreak > nul && start http://localhost:8000"

:: Inicia o Uvicorn
python -m uvicorn demo.server:app --host 127.0.0.1 --port 8000 --reload

pause
