#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
#set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1

if [ $# -ne 3 ]; then
#    NET=VGG_CNN_M_1024
    #NET=tattc
    NET=tattc_voc
else
    NET=$2
fi

if [ $# -ne 2 ]; then
    GPU_ID=0
else
    GPU_ID=$1
fi

NET_lc=${NET,,}
ITERS=100000
#ITERS=10
DATASET_TRAIN=tattc_voc_032816_tattoo
DATASET_TEST=tattc_voc_032816_test

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

#LOG="experiments/logs/${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/${NET}_`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#NET_INIT=data/imagenet_models/${NET}.v2.caffemodel
NET_INIT=data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel

#cmd="time ./tools/train_net.py --gpu ${GPU_ID} \
#  --solver models/${NET}/faster_rcnn_end2end/solver.prototxt \
#  --weights ${NET_INIT} \
#  --imdb ${DATASET_TRAIN} \
#  --iters ${ITERS} \
#  --cfg experiments/cfgs/faster_tattc.yml \
#  ${EXTRA_ARGS}"

cmd="time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/tattc_voc.yml \
  ${EXTRA_ARGS}"
echo $cmd
eval $cmd

