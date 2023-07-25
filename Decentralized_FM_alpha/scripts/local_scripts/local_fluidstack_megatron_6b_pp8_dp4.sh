#!/bin/bash
cd ~/GPT-home-private

MASTER_ADDR=$1
PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3

GPUS_PER_NODE=2
# Change for multinode config
MASTER_PORT=9000
NNODES=$4
NODE_RANK=$5
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

num_layers=$6
global_batch_size=$7

MICRO_BATCH_SIZE=4

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export GLOO_SOCKET_IFNAME=enp1s0
export NCCL_SOCKET_IFNAME=enp1s0

VOCAB_FILE=task_datasets/data/bert-large-cased-vocab.txt
TRAIN_FILE=task_datasets/data/QQP/train.tsv
VALID_FILE=task_datasets/data/QQP/dev.tsv
TEST_FILE=task_datasets/data/QQP/test.tsv

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
MODEL_ARGS="--num-layers $num_layers --hidden-size 4096 --num-attention-heads 32 --micro-batch-size $MICRO_BATCH_SIZE --global-batch-size $global_batch_size --seq-length 2048 --max-position-embeddings 2048"
PARALLEL_ARGS="--distributed-backend nccl --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE  --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE --DDP-impl local --no-bias-dropout-fusion"
NLP_ARGS="--tokenizer-type BertWordPieceLowerCase --vocab-file $VOCAB_FILE  --train-data-path $TRAIN_FILE  --valid-data-path $VALID_FILE  --test-data-path $TEST_FILE"
HYPER_PARA_ARGS="--optimizer sgd --lr 0.0001 --train-iters 3"
OPTION_ARGS="--fp16 --checkpoint-activations"
timestamp=$(date +%Y_%m_%d_%H_%M)

log_path="./logs/${timestamp}_megatron_gpt3_39b_w${NNODES}_t${TENSOR_PARALLEL_SIZE}_p${PIPELINE_PARALLEL_SIZE}"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS  ./dist_megatron_train_qqp.py $MODEL_ARGS $PARALLEL_ARGS $NLP_ARGS $HYPER_PARA_ARGS $OPTION_ARGS