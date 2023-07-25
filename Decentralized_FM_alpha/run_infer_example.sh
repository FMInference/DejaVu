PATH_TO_MODE="XXXX"
PATH_TO_INFER_DATA="YYYY"
NUM_LAYERS=4
MAX_LAYERS=32
# budget == max(#batch_size * #seq_length)
BUDGET=10000

ARGS="--model-name $PATH_TO_MODE \
--model-type opt \
--seed 42 \
--fp16 \
--num-layers $NUM_LAYERS \
--max-layers $MAX_LAYERS \
--budget $BUDGET \
--num-iters 10000000000 \
--dist-url tcp://127.0.0.1:9031 \
--token-micro-batch-size 2 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data $PATH_TO_INFER_DATA"

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
