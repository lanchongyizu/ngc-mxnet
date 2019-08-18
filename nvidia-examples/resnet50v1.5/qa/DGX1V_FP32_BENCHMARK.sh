#!/bin/bash
python ../benchmark.py --executable ../runner -n 1,4,8 -b 32,64,96 -i 100 -e 8 -w 4 --num-examples 12800 -o report.json --dtype=float32
python check_perf.py benchmark_baselines/RN50_mxnet_18.11-py3-stage_fp32.json report.json
