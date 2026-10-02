# 36-hour production run for the biggest 1660 Ti 6GB model (gpu9_max512).
# C=768 L=12 H=12 T=512, ~86M params, BPE-2000, chat + tool-format corpus.
# ~9000 optimizer steps x 8192 tokens ~= 74M tokens. Smoke measured ~12.5s/step
# (step 10 -> 20), so 9000 steps is ~31h and stays inside 36h.
#
# Run from an elevated PowerShell prompt in C:\dev\llm-gpu-9:
#   powershell -ExecutionPolicy Bypass -File .\setup\run_max512_36h.ps1
#
# Resume after interruption (weights-only checkpoints keep working):
#   .\venv\Scripts\python.exe train.py --resume --checkpoint output\checkpoints\gpu9_max512 `
#     --steps <remaining> --no-prompt
#
# Probe (50 prompts) and chat after/while training:
#   .\venv\Scripts\python.exe tools\n50_probe.py --checkpoint output\checkpoints\gpu9_max512
#   .\venv\Scripts\python.exe interactive.py --checkpoint output\checkpoints\gpu9_max512 --chat
#   .\venv\Scripts\python.exe tools\assistant_tools.py --checkpoint output\checkpoints\gpu9_max512

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..
.\venv\Scripts\python.exe train.py --config setup\gpu9_max512_config.json `
  --checkpoint output\checkpoints\gpu9_max512 `
  --steps 9000 --no-prompt
