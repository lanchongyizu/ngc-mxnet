# This script score ResNet50 checkpoint in FP32 on 1 GPUs using 64 batch size
# Usage ./SCORE_FP32.sh <model prefix> <epoch> <additionals flags>

./runner -n 1 -b 64 --dtype float32 --only-inference --model-prefix $1 --load-epoch $2 -e 1 ${@:3}
