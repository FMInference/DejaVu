cd ~/GPT-home-private

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=enp1s0

ip=$1
world_size=$2
rank=$3
cuda_id=$4
timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=central_ps

# Change the script here for different settings.
############################################################
num_layers=$5
ga_step=$6
batch_size=64
############################################################

if [ $# -eq 7 ]
then
  mode=$7
else
  mode='default'
fi

let "global_batch_size = $ga_step*$batch_size*6"
echo "Global Batch size: $global_batch_size, Num of Layer: $num_layers. Mode: $mode"

DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode gpipe --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 8 --data-group-size 6"
MODEL_CONF="--seq-length 2048 --embedding-dim 4096 --num-heads 32 --num-layers $num_layers --batch-size $batch_size --micro-batch-size 1"

python3 dist_training_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "./logs/${timestamp}_GPT3_6B_world32_rank${rank}_L${num_layers}_B${global_batch_size}_fluidStack_${mode}.log"
