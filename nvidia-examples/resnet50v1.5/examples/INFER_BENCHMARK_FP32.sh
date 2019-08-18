# This script launches ResNet50 inference benchmark in FP32 on 1 GPU with 1,2,4,32,64,96 batch size
# Usage ./INFER_BENCHMARK_FP32.sh <additionals flags>

python benchmark.py -n 1 -b 1,2,4,32,64,96 --only-inference -e 3 -w 1 -i 100 -o report.json $@
