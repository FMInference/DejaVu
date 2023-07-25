file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_c4_val_opt_66b_sparse.jsonl
eval_file=./c4_val/eval_c4_val_opt_66b_sparse.txt
    
echo "start running ${file}"

export SPRARSE_PATH=$PATH_TO_SPARSITY_PREDICTOR
export LAYER=54
export TOPK=5000
export ATTN_TOPK_1=36
export ATTN_TOPK_2=36
export SPARSE_ATT=1

LAYER=54
TOPK=5000
ATTN_TOPK_1=36
ATTN_TOPK_2=36

ARGS="--model-name /lustre/fsw/nvresearch/ldm/diffusion/checkpoint/opt-66b-new \
--model-type opt-ml-att-sparse \
--seed 42 \
--fp16 \
--num-layers 8 \
--max-layers 64 \
--budget 32800 \
--num-iters 2 \
--dist-url tcp://127.0.0.1:9032 \
--token-micro-batch-size 2 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

python -c "import json
import numpy as np

logprobs = []

with open('$output_file') as f:
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print('perplexity:', perplexity)" > $eval_file
cat $eval_file