#!/bin/bash
#BSUB -n 1                     # 1 cores
#BSUB -W 3:59                   # 3 hours run-time
#BSUB -R "rusage[mem=8000]"     # 8 GB per core
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10000]"
#BSUB -o /nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/exe_log/stable_diffusion.out.%J
#BSUB -e /nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/exe_log/stable_diffusion.err.%J
env2lmod
module load gcc/6.3.0 cuda/11.0.3             # Load modules from Euler setup
module load eth_proxy
source activate base                                     # Activate my conda python environment
cd /nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/GPT-home-private      # Change directory

nvidia-smi

ifconfig

COOR_CONF="--coordinator-server-ip 129.132.93.85"
#!/bin/bash

# export http_proxy=http://proxy.ethz.ch:3128
# export https_proxy=https://proxy.ethz.ch:3128
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1


python -u local_latency_inference_stable_diffussion.py $COOR_CONF \