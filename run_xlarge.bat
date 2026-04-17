@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
set PYTHONIOENCODING=utf-8
echo [x] CAA on 4B > xlarge.log
.venv\Scripts\python.exe -u src\compute_caa_vector.py --model xlarge --layers 13 18 23 >> xlarge.log 2>&1
echo [x] sweep on 4B >> xlarge.log
.venv\Scripts\python.exe -u src\sweep_layers.py --model xlarge --layers 13 18 23 --coefs 1.0 2.0 4.0 --n-prompts 10 >> xlarge.log 2>&1
echo [x] deepseek judge on 4B >> xlarge.log
.venv\Scripts\python.exe -u src\deepseek_judge.py --sweep data\sweep_qwen_xlarge.jsonl --out data\judge_deepseek_xlarge.jsonl >> xlarge.log 2>&1
echo DONE_XLARGE >> xlarge.log
