@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
echo [saes] starting SAE training large L14 >> chain.log
.venv\Scripts\python.exe -u src\train_sae.py --model large --layer 14 --tokens 1500000 --batch-size 4 --seq-len 128 >> chain.log 2>&1
echo [saes] SAE large done, starting SAE small L14 >> chain.log
.venv\Scripts\python.exe -u src\train_sae.py --model small --layer 14 --tokens 1500000 --batch-size 4 --seq-len 128 >> chain.log 2>&1
echo DONE_SAES >> chain.log
