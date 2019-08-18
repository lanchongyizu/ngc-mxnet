# This script launches ResNet50 benchmark in FP32 on 1,4,8 GPUs with 32,64,96 batch size
# Usage ./BENCHMARK_FP32.sh <additionals flags>

python benchmark.py -n 1,4,8 -b 32,64,96 -e 2 -w 1 -i 100 --dtype float32 -o report.json $@
