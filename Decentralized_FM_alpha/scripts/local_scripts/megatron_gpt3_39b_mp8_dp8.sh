#!/bin/bash
cd ~/GPT-home-private
source activate pytorch_p38

MICRO_BATCH_SIZE=$1
PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3


# Change for multinode config
# MASTER_ADDR=localhost
MASTER_ADDR=$4
MASTER_PORT=6000
GPUS_PER_NODE=$5
NNODES=$6
NODE_RANK=$7
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


VOCAB_FILE=glue_dataset/data/bert-large-cased-vocab.txt
TRAIN_FILE=glue_dataset/data/QQP/train.tsv
VALID_FILE=glue_dataset/data/QQP/dev.tsv
TEST_FILE=glue_dataset/data/QQP/test.tsv

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
MODEL_ARGS="--num-layers 48 --hidden-size 8192 --num-attention-heads 64 --micro-batch-size $MICRO_BATCH_SIZE --global-batch-size 1024 --seq-length 1024 --max-position-embeddings 1024"
PARALLEL_ARGS="--distributed-backend nccl --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE  --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE --DDP-impl local --no-bias-dropout-fusion"
NLP_ARGS="--tokenizer-type BertWordPieceLowerCase --vocab-file $VOCAB_FILE  --train-data-path $TRAIN_FILE  --valid-data-path $VALID_FILE  --test-data-path $TEST_FILE"
HYPER_PARA_ARGS="--optimizer sgd --lr 0.0001 --train-iters 1"
OPTION_ARGS="--fp16 --checkpoint-activations"
timestamp=$(date +%Y_%m_%d_%H_%M)

log_path="./logs/${timestamp}_megatron_gpt3_39b_w${NNODES}_t${TENSOR_PARALLEL_SIZE}_p${PIPELINE_PARALLEL_SIZE}"

if [ $# -eq 7 ]
then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS  ./dist_megatron_train_qqp.py $MODEL_ARGS $PARALLEL_ARGS $NLP_ARGS $HYPER_PARA_ARGS $OPTION_ARGS>> "${log_path}_default.log"
elif [ $# -eq 9 ]
then
  DELAY_MS=$7
  RATE_GBIT=$8
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python -m torch.distributed.launch $DISTRIBUTED_ARGS  ./dist_megatron_train_qqp.py $MODEL_ARGS $PARALLEL_ARGS $NLP_ARGS $HYPER_PARA_ARGS $OPTION_ARGS>> "${log_path}_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark megatron training is done."
