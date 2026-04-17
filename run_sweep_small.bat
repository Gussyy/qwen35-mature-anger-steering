@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
.venv\Scripts\python.exe -u src\sweep_layers.py --model small > sweep_small.log 2>&1
echo DONE_SWEEP_SMALL >> sweep_small.log
