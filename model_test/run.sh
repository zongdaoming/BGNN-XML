#!/bin/sh

set -x 

PARTITION=$1
GPUS=$2
config=$3

declare -u expname
expname=`basename ${config} .yaml`

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

currenttime=`date "+%Y%m%d%H%M%S"`
g=$(($2<8?$2:8))

mkdir -p  results/${expname}/train_log

srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=${expname} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=$g \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u -W ignore model_test.py \
    --config $3 \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log


#* sh ./run.sh test 1 ./configs/2e_5_acc.yaml


# sh ./run.sh test 1 ./configs/3e_5_acc.yaml
# sh ./run.sh test 1 ./configs/4e_5_acc.yaml
# sh ./run.sh test 1 ./configs/5e_5_acc.yaml

# sh ./run.sh test 1 ./configs/2e_6_acc.yaml
# sh ./run.sh test 1 ./configs/3e_6_acc.yaml
# sh ./run.sh test 1 ./configs/4e_6_acc.yaml
# sh ./run.sh test 1 ./configs/5e_6_acc.yaml



