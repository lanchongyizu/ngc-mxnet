#!/bin/bash
../runner -n 8 -b 208 --seed 0 --report report.json
python check_curves.py curve_baselines/JoC_RN50_FP16_curve_baseline.json report.json
