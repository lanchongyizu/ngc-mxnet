# This script launches ResNet50 benchmark in FP16 on 1,4,8 GPUs with 64,128,192,208 batch size
# Usage ./BENCHMARK_FP16.sh <additionals flags>

python benchmark.py -n 1,4,8 -b 64,128,192,208 -e 2 -w 1 -i 100 -o report.json $@
