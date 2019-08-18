# This script launches ResNet50 training in FP16 on 8 GPUs using 1664 batch size (208 per GPU)
# Usage ./RN50_FP16_8GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 8 -b 208 --model-prefix model ${@:2}
