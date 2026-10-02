@echo off
REM 36-hour production run for gpu9_max512. Command Prompt, no PowerShell.
REM C=768 L=12 H=12 T=512, ~86M params, 9000 steps (~31h at smoke speed).
REM Resume: venv\Scripts\python.exe train.py --resume --checkpoint output\checkpoints\gpu9_max512 --steps REMAINING --no-prompt
cd /d "%~dp0.."
venv\Scripts\python.exe auto_train.py --config setup\gpu9_max512_config.json --checkpoint output\checkpoints\gpu9_max512 --steps 9000 --no-prompt --prompt "User: Tell me a short story about a cat. Assistant:"
