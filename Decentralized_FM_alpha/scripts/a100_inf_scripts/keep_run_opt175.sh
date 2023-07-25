i=0
while :
do

  file=/data/queries_opt175/request_${i}.jsonl
  outfile=/data/queries_opt175/output_request_${i}.jsonl
  if [ -f "$file" ]; then
  if [ ! -f "$outfile" ]; then
      
      echo "start running ${file}"
  
      ARGS="--model-name /data/models/opt-175b-new \
      --model-type opt \
      --seed 42 \
      --fp16 \
      --num-layers 12 \
      --max-layers 96 \
      --budget 26800 \
      --num-iters 99999999999999 \
      --dist-url tcp://127.0.0.1:9031 \
      --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
      --pp-mode pipe_sync_sample_mask_token_pipe \
      --infer-data ${file}"

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
  fi
  fi

  ((i++))
  # Max id of jobs
  if [ "$i" -ge "500" ]; then
    echo "stopping submitting tasks"
    break
  fi
  sleep 0.01
done
