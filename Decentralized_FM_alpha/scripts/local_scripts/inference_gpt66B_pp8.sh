cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3
cuda_id=$4
timestamp=$(date +%Y_%m_%d_%H_%M)

token_mbs=2
DIST_CONF="--pp-mode pipe_sync_greedy_token_pipe --world-size $world_size --pipeline-group-size $world_size --data-group-size 1 --rank $rank  --cuda-id $cuda_id"
MODEL_CONF="--model-type gptj --model-name ./pretrained_debug_models/gpt-j-66B"
INFERENCE_CONF="--batch-size 20 --input-seq-length 1024 --generate-seq-length 100 --micro-batch-size 1 --num-layers 8  --token-micro-batch-size $token_mbs"


if [ "$world_size" -ne 8 ]
then
  echo "Not correct number of nodes"
  exit 1
fi


python dist_inference_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF $INFERENCE_CONF>> "./logs/${timestamp}_token_${token_mbs}_rank_${rank}_inference_66B_pp8_default.log"
