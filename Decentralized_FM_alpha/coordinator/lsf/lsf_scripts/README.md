# Run GPT@Home on Euler HPC cluster in ETH Zurich


- Check submitted job states:

      bjobs

Go to this directory in the Euler cluster:

- Load Lib test:

      bsub < lsf_foo.bsub

- Local training test:

      bsub < lsf_local_train_test.bsub


- The current repo in the Euler cluster:

      /nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/GPT-home-private 



- To check what is going on in a node,

       bjob_connect job_ID

- Kill all jobs:

       bkill 0

- New commands:

  - Run coordinator server:
        
        python lsf_coordinate_server.py --coordinator-server-ip 129.132.93.85 --coordinator-type inference

  - Run job scheduler:

        python lsf_job_scheduler.py --coordinator-server-ip 129.132.93.85

  - Submit jobs:
  
        python job_submit_client.py --coordinator-server-ip 129.132.93.85 --submit-job inference --job-name lsf_latency_stable_diffusion
        python job_submit_client.py --coordinator-server-ip 129.132.93.85 --submit-job inference --job-name lsf_latency_gpt_j_6B