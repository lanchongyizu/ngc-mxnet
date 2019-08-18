# This script launches ResNet50 training in FP16 on 4 GPUs using 832 batch size (208 per GPU)
# Usage ./RN50_FP16_4GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 4 -b 208 --model-prefix model ${@:2}
