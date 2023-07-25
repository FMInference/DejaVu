cd ~/GPT-home-private
# source activate pytorch_p38
ip=$1
world_size=$2
pipeline_size=$3
cpu_size=$(($world_size-$pipeline_size))
rank=$4
node_type=$5

stage_num_layers=4
global_num_layers=$(($stage_num_layers*$pipeline_size))

producer_buffer_size=30
consumer_buffer_size=2
micro_batch_size=1

input_seq_length=1900
generate_seq_length=100

timestamp=$(date +%Y_%m_%d_%H_%M)

if [ $node_type == "GPU" ]
then
  DIST_CONF="--pp-mode pipe_hybrid_greedy_async --world-size $world_size --pipeline-group-size $pipeline_size --rank $rank --node-type $node_type --use-cuda True"
else
  DIST_CONF="--pp-mode pipe_hybrid_greedy_async --world-size $world_size --pipeline-group-size $pipeline_size --rank $rank --node-type $node_type --use-cuda False"
fi

MODEL_CONF="--model-type gptj --model-name ./pretrained_debug_models/gpt-j-175B"
INFERENCE_CONF="--num-iters 2 --input-seq-length $input_seq_length --generate-seq-length $generate_seq_length --prompt-micro-batch-size $micro_batch_size --token-micro-batch-size $micro_batch_size --stage-num-layers $stage_num_layers --global-num-layers $global_num_layers"
BUF_CONF="--producer-buffer-size $producer_buffer_size --consumer-buffer-size $consumer_buffer_size"

if [ "$world_size" -ne 54 ]
then
  echo "Not correct number of nodes"
  exit 1
fi

log_name=${timestamp}_175b_hybrid_inference_sq${input_seq_length}_${generate_seq_length}_gpu${pipeline_size}cpu${cpu_size}_pb${producer_buffer_size}_cb${consumer_buffer_size}_mbs${micro_batch_size}_default.log

python3 dist_inference_hybrid_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF $INFERENCE_CONF $BUF_CONF  >> "./logs/$log_name"