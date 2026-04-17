@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
set PYTHONIOENCODING=utf-8
echo [dsj] judging 2B sweep > dsj.log
.venv\Scripts\python.exe -u src\deepseek_judge.py --sweep data\sweep_qwen_large.jsonl --out data\judge_deepseek_large.jsonl >> dsj.log 2>&1
echo [dsj] judging 0.8B sweep >> dsj.log
.venv\Scripts\python.exe -u src\deepseek_judge.py --sweep data\sweep_qwen_small.jsonl --out data\judge_deepseek_small.jsonl >> dsj.log 2>&1
echo [dsj] done >> dsj.log
