#!/bin/bash
../runner -n 8 -b 96 --dtype float32 --seed 0 --report report.json
python check_curves.py curve_baselines/JoC_RN50_FP32_curve_baseline.json report.json
