cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3
timestamp=$(date +%Y_%m_%d_%H_%M)

DIST_CONF="--pp-mode pipe_sync_greedy --world-size $world_size --pipeline-group-size $world_size --data-group-size 1 --rank "$rank""
MODEL_CONF="--model-type gptj --model-name ./pretrained_debug_models/gpt-j-6B"
INFERENCE_CONF="--batch-size 48 --input-seq-length 512 --generate-seq-length 32 --micro-batch-size 1 --num-layers 2"


if [ "$world_size" -ne 14 ]
then
  echo "Not correct number of nodes"
  exit 1
fi


python dist_inference_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF $INFERENCE_CONF>> "./logs/${timestamp}_GPTJ_inference_pp14_default.log"