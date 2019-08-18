# This script launches ResNet50 training in FP16 on 1 GPUs using 208 batch size (208 per GPU)
# Usage ./RN50_FP16_1GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 1 -b 208 --model-prefix model ${@:2}
