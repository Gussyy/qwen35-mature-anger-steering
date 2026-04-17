@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
.venv\Scripts\python.exe -u src\compute_caa_vector.py --model small > caa_small.log 2>&1
echo DONE_CAA_SMALL >> caa_small.log
