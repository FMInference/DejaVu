coordinator_server_ip=$1

cd ~/GPT-home-private

num_gpu=8

ARGS="--model-type gptj --model-name ./pretrained_debug_models/gpt-j-175B \
--coordinator-server-ip $coordinator_server_ip
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 96 \
--budget 26800 \
--num-iters 3 \
--dist-url tcp://127.0.0.1:9031 \
--world-size ${num_gpu} --pipeline-group-size ${num_gpu} --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe"


for (( i=0; i<${num_gpu}; i++))
do
    python3 multi_gpu_inference_w_crusoe_coordinator.py ${ARGS} --cuda-id $i --rank $i &
done

wait


