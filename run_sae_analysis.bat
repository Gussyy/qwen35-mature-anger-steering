@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
echo [sae_ana] starting large analysis >> chain.log
.venv\Scripts\python.exe -u src\sae_analysis.py --model large --layer 14 --coef 2.0 --n-prompts 30 --top-k 30 >> chain.log 2>&1
echo [sae_ana] large done, starting small analysis >> chain.log
.venv\Scripts\python.exe -u src\sae_analysis.py --model small --layer 14 --coef 1.0 --n-prompts 30 --top-k 30 >> chain.log 2>&1
echo DONE_SAE_ANALYSIS >> chain.log
