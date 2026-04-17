@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
.venv\Scripts\python.exe -u src\sweep_layers.py --model large > sweep_large.log 2>&1
echo DONE_SWEEP_LARGE >> sweep_large.log
