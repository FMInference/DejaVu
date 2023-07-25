cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3
cuda_id=$4
timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=central_ps

# Change the script here for different settings.
############################################################
ga_step=$5
num_layers=$6
batch_size=$7
############################################################

let "global_batch_size = $ga_step*$batch_size*8"

DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode gpipe --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 3 --data-group-size 1"
MODEL_CONF="--embedding-dim 2048 --num-heads 16 --num-layers $num_layers --batch-size $batch_size --micro-batch-size 1"

if [ "$world_size" -ne 3 ]
then
  echo "Not correct number of nodes"
  exit 1
fi

log_mode=$8
log_path="./logs/${timestamp}_gpipe_gpt3_xl_pp3_l${num_layers}_b${global_batch_size}_rank${rank}_${log_mode}"

if [ $# -eq 8 ]
then
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "${log_path}_default.log"
elif [ $# -eq 9 ]
then
  case=$9
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/heterogeneous_setup_case"$case".sh
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "${log_path}_heter${case}.log"
  sh ./scripts/tc_scripts/clear.sh
elif [ $# -eq 10 ]
then
  DELAY_MS=$9
  RATE_GBIT=${10}
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF --trace-postfix "d${DELAY_MS}b${RATE_GBIT}" >> "${log_path}_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark training is done."