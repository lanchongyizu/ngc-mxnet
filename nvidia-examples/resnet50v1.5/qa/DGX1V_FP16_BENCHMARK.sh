#!/bin/bash
python ../benchmark.py --executable ../runner -n 1,4,8 -b 64,128,192,208 -i 100 -e 8 -w 4 --num-examples 25600 -o report.json
python check_perf.py benchmark_baselines/RN50_mxnet_18.11-py3-stage_fp16.json report.json
