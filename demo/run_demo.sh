#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.."

echo "========================================================"
echo "  RAG Multimodal Enterprise - Inicializando Demo Local"
echo "========================================================"
echo ""

if [ -f ".venv/bin/activate" ]; then
    echo "[*] Ativando ambiente virtual .venv..."
    source .venv/bin/activate
fi

echo "[*] Iniciando servidor FastAPI em http://localhost:8000 ..."
echo "[*] Pressione CTRL+C para encerrar o servidor."
echo ""

# Tenta abrir o navegador em background
(sleep 2 && (xdg-open http://localhost:8000 || open http://localhost:8000 || sensible-browser http://localhost:8000)) 2>/dev/null &

python -m uvicorn demo.server:app --host 127.0.0.1 --port 8000 --reload
