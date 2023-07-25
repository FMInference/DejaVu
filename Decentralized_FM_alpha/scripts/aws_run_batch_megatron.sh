case=$1
#batches=(1024 2048 4096)
#layers=(24 32 40)

#for layer in "${layers[@]}"
#do
#  for batch in "${batches[@]}"
#  do
#    echo "$batch, $layer"
#    # bash aws_run_megatron_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 "$layer" "$batch" "$case"
#    bash aws_run_megatron_scheduled_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 "$layer" "$batch" "$case"
#    sleep 5s
#    bash copy_rank_last_logs.sh 54.218.215.178
#    sleep 3s
#  done
#done

bash aws_run_megatron_scheduled_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 24 1024 "$case"
sleep 10s

bash aws_run_megatron_scheduled_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 32 1024 "$case"
sleep 10s

bash aws_run_megatron_scheduled_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 40 1024 "$case"
sleep 10s

