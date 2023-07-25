# Global Coordinator

## Set up coordinator 

- Install couchdb:

        pip install pycouchdb

- So far just use Xiaozhe's DB service, we can seperated ones later: 
       
        http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/

- Doc about couchdb:

        https://github.com/histrio/py-couchdb/blob/master/docs/source/quickstart.rst
        

## Format of Job Submission File

- job_type_info: (latency_inference, batch_inference)
- job_state: (job_queued, job_running, job_finished, job_returned)
- time:
  - job_queued_time:
  - job_start_time:
  - job_end_time:
  - job_returned_time:
- task_api:
  - inputs:
  - model_name: gpt_j_6B, stable_diffusion. 
  - task_type: seq_generation, image_generation
  - parameters:
    - max_new_tokens
    - return_full_text
    - do_sample
    - temperature
    - top_p
    - max_time
    - num_return_sequences
    - use_gpu
    - (etc.)
  - outputs: 

        
## Format of HeartBeat Submission File

- task_type:
- model_name:
- cluster_location:
- last_heartbeat_time:
        

