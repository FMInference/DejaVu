# GPT-home-private

## Setup:

- Use AWS Deep Learning Base AMI


- Install PyTorch env (old): 

      pip3 install torch==1.9.0+cu111 torchtext -f https://download.pytorch.org/whl/torch_stable.html
      pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
      pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

      # Magic, not sure why cupy-cuda111 would not work, it seems that cupy-cuda111 will use different PTX from torch.
      pip3 install cupy-cuda111==8.6.0
      pip3 install transformers

- Install PyTorch env (latest):
      
      pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
      pip3 install cupy-cuda11x==11.0.0
      python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
      pip3 install transformers

- Install PyTorch env (CPU-latest):

      pip3 install --pre torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

- Install deepspeed for some micro-benchmark (optional)

      pip install deepspeed

- Clone this repo:
        
      git clone https://github.com/BinhangYuan/GPT-home-private.git

- set the github cache (Optional):

      git config credential.helper 'cache --timeout=30000'

- Download a tiny dataset:

      wget https://binhang-language-datasets.s3.us-west-2.amazonaws.com/glue_qqp_dataset/data.tar.xz -P ./glue_dataset/
      
      tar -xvf ./glue_dataset/data.tar.xz -C ./glue_dataset/

- Setup network configuration:

      export GLOO_SOCKET_IFNAME=ens3

      export NCCL_SOCKET_IFNAME=ens3


- Use TC scripts to control network delay and bandwidth:
  

## Run Distributed Gpipe:

- On each node, run:
      
      python dist_pipeline_runner.py --dist-url tcp://XXX.XXX.XXX.XXX:9000 --world-size N --rank i (i=0,...,N-1)



## Run deepspeed benchmark:

- Update public-ip and hostname in the ./scripts/ip_list.sh file

- Update the host name with slots (number of GPUs) in ./scripts/ds_hostnames.sh file

- Sync code to all nodes

- Setup password free ssh cluster by executing (under the ./script/ directory):

      bash ssh_pass_free.sh

- SSH to the rank-0 node, on that node run:

      source activate pytorch_p38

- A sample run:

      deepspeed --hostfile=./scripts/ds_hostnames dist_deepspeed_zero_s3.py --embedding-dim 2048 --seq-length 2048 --batch-size 1024 --num-layers 40 --micro-batch-size 4

- Batch run all settings:

       cd ./scripts
       bash local_run_deepspeed_batch_on_rank0.sh #CASE


  
## Run with Advanced Scripts (under scripts directory):

- First update the public IPs and private IP of the rank-0 node in ip_list.sh.

- Allow SSH connects: 

      bash accept_ssh_keys.sh

- Update local repository:

      bash aws_sync_code.sh #GIT_TOKEN
      
- Enable environment: (This is optional but load conda env seems to be slow for the first time)

      bash aws_foo_load_lib.sh

- Setup heterogeneous network (update the private IPs in ./scheduler/generate_heterogeneous_tc.py, sync the code to AWS!):

      bash aws_generate_heter_tc.sh #HETER_CASE (3/4/5)

- Optional(Play with it by starting TC, this is not needed if you use the next block of cmds to start benchmarks):

      bash aws_start_heter_tc.sh #HETER_CASE (3/4/5)

- Run Schedulers (under scheduler/heuristic_evolutionary_solver directory) to get assignments and estimated cost

      python scheduler.py

- Update correspond aws_run_gpt3_*_training.sh with the above output:
- Run Tasks (e.g.,):

      bash aws_run_gpt3_training.sh gpt3_small_pp3_dp4.sh
      bash aws_run_gpt3_training.sh gpt3_small_pp3_dp4.sh #HETER_CASE
      bash aws_run_gpt3_training.sh gpt3_small_pp3_dp4.sh #DELAY #BANDWIDTH


- Clear logs:

      bash aws_clear_logs.sh

- Copy training logs from Rank-0 node (For my implementation the benchmark result is on the rank-0 node.)

      bash copy_rank0_logs.sh

- Download and generate trace:

      bash copy_traces.sh #PREFIX
      bash generate_traces.sh #PREFIX

## Run the System on Euler with Coordinator

- First log in to ETH Zurich Euler cluster (need to use VPN if not on campus), our working directory:

      /nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/

- Get the Euler ssh client's node IP, and start the coordinator server (under ./coordinator directory), e.g.:

      python coordinate_server.py --coordinator-server-ip 129.132.93.88 --coordinator-type inference/train

- Alternatively, we can use the following script to submit multiple jobs:

      bash multi_inference_jobs_submit.sh 129.132.93.88 lsf_gptJ_inf_4RTX2080Ti 3 10

- On a different terminal, start a job submission client to submit a job:

      python job_submit_client.py --coordinator-server-ip 129.132.93.88 --submit-job inference --job-name lsf_gptJ_inf_4RTX2080Ti

- So far, we need to manually change the IP in the job-submit templates, e.g. change lsf_gptJ_inf_4gpu.bsub file.

- Check submitted job states:

      bjobs







