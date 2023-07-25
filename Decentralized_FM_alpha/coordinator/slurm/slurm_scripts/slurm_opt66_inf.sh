#!/bin/bash
#SBATCH --job-name=gptneox
#
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:4
#SBATCH --time=3:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/exe_log/gptneox_%j.log
port=$1

source activate base                          # Activate my conda python environment
cd /afs/cs.stanford.edu/u/biyuan/GPT-home-private     # Change directory

nvidia-smi

ls -l /sys/class/net/
netif=`echo "import os; print(sorted([x for x in os.listdir('/sys/class/net/') if x.startswith('en')])[0])" | python`
echo setting network interface to $netif
export NCCL_SOCKET_IFNAME=$netif
export GLOO_SOCKET_IFNAME=$netif
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

world_size=32
machine_size=8
n_gpu_per_machine=$((world_size/machine_size))

i=0
while :
do
  if [ "$i" -ge "$n_gpu_per_machine" ]; then
      break
  fi
  
  DIST_CONF="--pp-mode pipe_sync_sample_mask_token_pipe --pipeline-group-size $world_size --cuda-id $i"
  MODEL_CONF="--model-type opt --model-name /sailhome/biyuan/scratch/models/opt-66b-new --num-iters 10"
  INFERENCE_CONF="--fp16  --budget 10400 --batch-size 24 --input-seq-length 512 --generate-seq-length 32 --micro-batch-size 1 --num-layers 2 --max-layers 64"
  COOR_CONF="--coordinator-server-ip 10.79.12.70  --unique-port $port"

  python -u dist_inference_runner_w_slurm_coordinator.py $DIST_CONF $MODEL_CONF $INFERENCE_CONF $COOR_CONF &

  ((port++))
  ((i++))

done

wait
