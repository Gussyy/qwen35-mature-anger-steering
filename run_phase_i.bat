@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
set PYTHONIOENCODING=utf-8
echo [pi] sae feature steer > phase_i.log
.venv\Scripts\python.exe -u src\sae_feature_steer.py --feat-ids 79 973 1206 1367 137 --alphas 1.0 2.0 4.0 --n-prompts 15 >> phase_i.log 2>&1
echo [pi] multi-layer >> phase_i.log
.venv\Scripts\python.exe -u src\multi_layer_steer.py --alphas 0.5 1.0 1.5 2.0 --n-prompts 15 >> phase_i.log 2>&1
echo [pi] anchor prompt >> phase_i.log
.venv\Scripts\python.exe -u src\anchor_plus_steer.py --coef 0.5 --n-prompts 15 >> phase_i.log 2>&1
echo DONE_PHASE_I >> phase_i.log
