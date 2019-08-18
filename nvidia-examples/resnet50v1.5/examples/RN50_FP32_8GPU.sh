# This script launches ResNet50 training in FP32 on 8 GPUs using 768 batch size (96 per GPU)
# Usage ./RN50_FP32_8GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 8 -b 96 --dtype float32 --model-prefix model ${@:2}
