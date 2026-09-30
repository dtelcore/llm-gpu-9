# ==============================================================================
# CUDA & PyCUDA Environment Setup Script for GTX 1660 Ti (CC 7.5)
# Target Context: llm-gpu-9
# Run this script from an elevated PowerShell prompt (Run as Administrator).
# ==============================================================================

# --- Ensure Administrator Privileges ---
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsInBuiltRole]::Administrator)) {
    Write-Host "This script requires Administrator privileges to modify system files and installers." -ForegroundColor Yellow
    Write-Host "Relaunching as Administrator..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    Exit
}

# --- Dynamic Path Configuration ---
# Automatically anchors to your current "llm gpu 5" main directory context
$BASE_DIR      = Get-Item .
$VENV_PATH     = Join-Path $BASE_DIR "venv"
$VS_INSTALLER  = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$MSVC_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
$VCVARSALL = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

Write-Host "Working Directory Context: $($BASE_DIR.FullName)" -ForegroundColor Cyan
Write-Host "Target Venv Destination:   $VENV_PATH" -ForegroundColor Cyan

# --- Step 1: Initialize Python Virtual Environment ---
Write-Host "`n=== STEP 1: Setting up Python Virtual Environment ===" -ForegroundColor Cyan
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Creating clean Python venv in $VENV_PATH..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Ensure Python 3.12 is available in your system path."
        Exit
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists at target location." -ForegroundColor DarkGreen
}

# --- Step 2: Verify MSVC 14.51 (VS 2026 Build Tools) ---
Write-Host "`n=== STEP 2: Verifying MSVC 14.51 ===" -ForegroundColor Cyan
if (-not (Test-Path "$MSVC_PATH\bin\Hostx64\x64\cl.exe")) {
    Write-Error "cl.exe not found under $MSVC_PATH. Install Visual Studio Build Tools 2026 with the MSVC v145 toolset."
    Exit
}
Write-Host "MSVC 14.51 already present." -ForegroundColor Green

# --- Step 3: Verify CUDA 13.2 nvcc (do not patch toolkit headers) ---
Write-Host "`n=== STEP 3: Verifying CUDA 13.2 ===" -ForegroundColor Cyan
if (-not (Test-Path "$CUDA_PATH\bin\nvcc.exe")) {
    Write-Error "nvcc not found at $CUDA_PATH\bin\nvcc.exe."
    Exit
}
Write-Host "CUDA 13.2 nvcc present. Toolkit headers are left untouched." -ForegroundColor Green

# --- Step 5: Environment Activation & MSVC Exporting ---
Write-Host "`n=== STEP 5: Activating Python Venv & Compiling Environment ===" -ForegroundColor Cyan
$activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Error "Virtual environment activation script missing!"
    Exit
}

# Use a cmd wrapper block to parse native MSVC variables into PowerShell env
$vcvarsall = $VCVARSALL
if (Test-Path $vcvarsall) {
    $setup_cmd = "@echo off`ncall `"$vcvarsall`" x64 -vcvars_ver=14.51`nset"
    $temp_batch = "$env:TEMP\setup_msvc_vars.bat"
    $setup_cmd | Out-File -FilePath $temp_batch -Encoding ASCII
    
    $env_output = & cmd /c "$temp_batch"
    foreach ($line in $env_output) {
        if ($line -match "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$") {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
    Remove-Item $temp_batch -Force -ErrorAction SilentlyContinue
    Write-Host "MSVC 14.51 variables exported to session." -ForegroundColor Green
}

# --- Step 6: Path Enforcement Strategy ---
Write-Host "`n=== STEP 6: Enforcing Path Dominance for CUDA 13.2 ===" -ForegroundColor Cyan
$env:CUDA_PATH = $CUDA_PATH
$env:PATH = "$CUDA_PATH\bin;$MSVC_PATH\bin\Hostx64\x64;$env:PATH"

# Set Distutils flags for Python compilation
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"

# --- Step 7: Core Dependencies & PyCUDA Build ---
Write-Host "`n=== STEP 7: Upgrading Pip Tools & Installing Pre-requisites ===" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# CRITICAL FIX: Install numpy first so the pycuda setup script can locate its C-headers
Write-Host "Installing numpy (required for PyCUDA compilation headers)..." -ForegroundColor Yellow
pip install numpy

Write-Host "Installing PyCUDA for Python 3.12..." -ForegroundColor Yellow
pip install pycuda --no-cache-dir 2>&1 | Tee-Object -FilePath "pycuda_build.log"