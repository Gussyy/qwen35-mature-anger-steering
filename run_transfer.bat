@echo off
cd /d E:\AgentCompany\Projects\LLmStreering
set PYTHONPATH=src
echo [transfer] starting transfer eval >> chain.log
.venv\Scripts\python.exe -u src\transfer_vector.py --l-large 14 --l-small 14 --c-small 1.0 >> chain.log 2>&1
echo [transfer] transfer done >> chain.log
.venv\Scripts\python.exe -u src\compare_sizes.py >> chain.log 2>&1
echo DONE_TRANSFER >> chain.log
