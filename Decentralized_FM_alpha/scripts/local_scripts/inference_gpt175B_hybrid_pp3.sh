cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3
timestamp=$(date +%Y_%m_%d_%H_%M)

token_mbs=128
DIST_CONF="--pp-mode pipe_hybrid_greedy --world-size $world_size --pipeline-group-size $world_size --data-group-size 1 --rank $rank  --cuda-id 0"
MODEL_CONF="--model-type gptj --model-name ./pretrained_debug_models/gpt-j-175B  --num-iters 3"
INFERENCE_CONF="--batch-size 128 --input-seq-length 1024 --generate-seq-length 100 --prompt-micro-batch-size 1 --num-layers 2 --token-micro-batch-size $token_mbs"

export NCCL_SOCKET_IFNAME=ens5
export GLOO_SOCKET_IFNAME=ens5

if [ "$world_size" -ne 3 ]
then
  echo "Not correct number of nodes"
  exit 1
fi

python dist_inference_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF $INFERENCE_CONF >> "./logs/${timestamp}_token_${token_mbs}_rank_${rank}_inference_hybrid_175B_pp3_default.log"
