# This script score ResNet50 checkpoint in FP16 on 1 GPUs using 128 batch size
# Usage ./SCORE_FP16.sh <model prefix> <epoch> <additionals flags>

./runner -n 1 -b 128 --only-inference --model-prefix $1 --load-epoch $2 -e 1 ${@:3}
