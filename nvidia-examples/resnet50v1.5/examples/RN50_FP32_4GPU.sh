# This script launches ResNet50 training in FP32 on 4 GPUs using 384 batch size (96 per GPU)
# Usage ./RN50_FP32_4GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 4 -b 96 --dtype float32 --model-prefix model ${@:2}
