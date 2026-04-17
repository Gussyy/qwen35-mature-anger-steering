@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
echo [chain] starting CAA small > chain.log
.venv\Scripts\python.exe -u src\compute_caa_vector.py --model small >> chain.log 2>&1
echo [chain] CAA small done, starting sweep small >> chain.log
.venv\Scripts\python.exe -u src\sweep_layers.py --model small >> chain.log 2>&1
echo [chain] sweep small done >> chain.log
echo DONE_CHAIN_SMALL >> chain.log
