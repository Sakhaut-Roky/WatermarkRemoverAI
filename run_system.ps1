# ==============================================================================
# WatermarkRemoverAI - Windows PowerShell System Bootstrapper
# ==============================================================================

$ErrorActionPreference = "Stop"
$WorkingDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $WorkingDir

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "    ✨ Initializing WatermarkRemoverAI Platform Engine           " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Detect Python
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    Write-Error "[-] Python interpreter not found on system PATH."
    exit 1
}

Write-Host "[*] Python executable: $PythonExe" -ForegroundColor Green

# 2. Virtual Environment Setup
$VenvDir = Join-Path $WorkingDir "venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[*] Creating dedicated virtual environment in $VenvDir..." -ForegroundColor Yellow
    & $PythonExe -m venv $VenvDir
} else {
    Write-Host "[*] Existing virtual environment detected at $VenvDir." -ForegroundColor Green
}

# 3. Activate Virtual Environment
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "[-] Virtual environment python.exe not found at $VenvPython"
    exit 1
}

# 4. Dependency Installation
Write-Host "[*] Upgrading pip and verifying package dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --quiet

if (Test-Path "requirements.txt") {
    Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & $VenvPython -m pip install -r requirements.txt
}

# 5. Service Orchestration
Write-Host "------------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "[*] Launching WatermarkRemoverAI Microservices..." -ForegroundColor Cyan

# Start FastAPI server on port 8000
Write-Host "[1/2] Starting FastAPI Core Platform (0.0.0.0:8000)..." -ForegroundColor Green
$FastApiProcess = Start-Process -FilePath $VenvPython -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info" -PassThru

Start-Sleep -Seconds 3

# Start Gradio QA Console on port 7860
Write-Host "[2/2] Starting Standalone Gradio QA Console (127.0.0.1:7860)..." -ForegroundColor Green
$GradioProcess = Start-Process -FilePath $VenvPython -ArgumentList "ui/app.py" -PassThru

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "    🚀 WatermarkRemoverAI Platform is ONLINE & OPERATIONAL       " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  • REST API & Documentation : http://localhost:8000/docs         " -ForegroundColor White
Write-Host "  • Alternative API Docs     : http://localhost:8000/redoc        " -ForegroundColor White
Write-Host "  • Standalone QA Console    : http://127.0.0.1:7860              " -ForegroundColor White
Write-Host "  • Embedded Web Interface   : http://localhost:8000/ui           " -ForegroundColor White
Write-Host "  • Health & Device Probe    : http://localhost:8000/api/v1/health" -ForegroundColor White
Write-Host "==================================================================" -ForegroundColor Cyan

try {
    Wait-Process -Id $FastApiProcess.Id, $GradioProcess.Id
} finally {
    Write-Host "[*] Stopping microservices..." -ForegroundColor Yellow
    Stop-Process -Id $FastApiProcess.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $GradioProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[*] All WatermarkRemoverAI services stopped." -ForegroundColor Green
}
