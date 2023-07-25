cd ~/GPT-home-private
# source activate pytorch_p38
ip=$1
world_size=$2
pipeline_size=$3
rank=$4
node_type=$5

stage_num_layers=4
global_num_layers=$(($stage_num_layers*$pipeline_size))

timestamp=$(date +%Y_%m_%d_%H_%M)

if [ $node_type == "GPU" ]
then
  DIST_CONF="--pp-mode pipe_hybrid_greedy_async --world-size $world_size --pipeline-group-size $pipeline_size --rank $rank --node-type $node_type --use-cuda True"
else
  DIST_CONF="--pp-mode pipe_hybrid_greedy_async --world-size $world_size --pipeline-group-size $pipeline_size --rank $rank --node-type $node_type --use-cuda False"
fi

MODEL_CONF="--model-type gptj --model-name ./pretrained_models/gpt-j-175B"
INFERENCE_CONF="--num-iters 3 --input-seq-length 512 --generate-seq-length 32 --prompt-micro-batch-size 1 --token-micro-batch-size 1 --stage-num-layers $stage_num_layers --global-num-layers $global_num_layers"
BUF_CONF="--producer-buffer-size 8 --consumer-buffer-size 4"

if [ "$world_size" -ne 8 ]
then
  echo "Not correct number of nodes"
  exit 1
fi


python3 dist_inference_hybrid_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF $INFERENCE_CONF $BUF_CONF  >> "./logs/${timestamp}_GPTJ_hybrid_inference_gpu3cpu2_default.log"