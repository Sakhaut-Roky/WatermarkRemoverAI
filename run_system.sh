#!/usr/bin/env bash
# ==============================================================================
# WatermarkRemoverAI - System Bootstrapper & Service Orchestrator
# ==============================================================================
# This script:
# 1. Detects or creates a Python virtual environment (.venv / venv)
# 2. Installs required dependencies from requirements.txt
# 3. Simultaneously orchestrates:
#    - FastAPI RESTful Server (Port 8000)
#    - Gradio QA Console (Port 7860)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "    ✨ Initializing WatermarkRemoverAI Platform Engine           "
echo "=================================================================="

# 1. Detect Python Interpreter
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v py &>/dev/null; then
    PYTHON_CMD="py -3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python interpreter not found on system PATH."
    exit 1
fi

echo "[*] Detected Python: $($PYTHON_CMD --version)"

# 2. Virtual Environment Setup
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating dedicated virtual environment in ./${VENV_DIR}..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo "[*] Existing virtual environment detected at ./${VENV_DIR}."
fi

# 3. Activate Virtual Environment (Support for Windows Git Bash & POSIX)
if [ -f "${VENV_DIR}/Scripts/activate" ]; then
    # Windows / Git Bash / MSYS2
    source "${VENV_DIR}/Scripts/activate"
elif [ -f "${VENV_DIR}/bin/activate" ]; then
    # POSIX / Linux / macOS / WSL
    source "${VENV_DIR}/bin/activate"
else
    echo "[-] Error: Virtual environment activation script missing."
    exit 1
fi

echo "[*] Virtual environment activated: $(which python)"

# 4. Dependency Installation
echo "[*] Upgrading pip and verifying package dependencies..."
python -m pip install --upgrade pip --quiet

if [ -f "requirements.txt" ]; then
    echo "[*] Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt
else
    echo "[-] Warning: requirements.txt not found. Skipping installation."
fi

# 5. Service Orchestration
echo "------------------------------------------------------------------"
echo "[*] Launching WatermarkRemoverAI Microservices..."

# Graceful cleanup on termination
cleanup() {
    echo ""
    echo "[*] Graceful shutdown sequence initiated..."
    if [ -n "$FASTAPI_PID" ]; then
        kill "$FASTAPI_PID" 2>/dev/null || true
    fi
    if [ -n "$GRADIO_PID" ]; then
        kill "$GRADIO_PID" 2>/dev/null || true
    fi
    echo "[*] All WatermarkRemoverAI services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start Service A: FastAPI Platform Gateway (Port 8000)
echo "[1/2] Starting FastAPI Core Platform (0.0.0.0:8000)..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &
FASTAPI_PID=$!

# Brief pause to allow FastAPI startup
sleep 3

# Start Service B: Standalone Gradio QA Console (Port 7860)
echo "[2/2] Starting Standalone Gradio QA Console (127.0.0.1:7860)..."
python ui/app.py &
GRADIO_PID=$!

echo "=================================================================="
echo "    🚀 WatermarkRemoverAI Platform is ONLINE & OPERATIONAL       "
echo "=================================================================="
echo "  • REST API & Documentation : http://localhost:8000/docs         "
echo "  • Alternative API Docs     : http://localhost:8000/redoc        "
echo "  • Standalone QA Console    : http://127.0.0.1:7860              "
echo "  • Embedded Web Interface   : http://localhost:8000/ui           "
echo "  • Health & Device Probe    : http://localhost:8000/api/v1/health"
echo "=================================================================="
echo "  Press Ctrl+C to terminate all services gracefully.             "
echo "=================================================================="

# Wait for background processes
wait "$FASTAPI_PID" "$GRADIO_PID"
