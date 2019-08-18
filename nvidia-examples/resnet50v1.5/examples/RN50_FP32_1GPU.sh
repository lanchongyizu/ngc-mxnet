# This script launches ResNet50 training in FP32 on 1 GPUs using 96 batch size (96 per GPU)
# Usage ./RN50_FP32_1GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 1 -b 96 --dtype float32 --model-prefix model ${@:2}
