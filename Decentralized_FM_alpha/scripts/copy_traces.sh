source ./ip_list.sh

# Current valid prefix includes:
# gpt3_gpipe_b64_1_l2048_m768_w3_p3_d1,
# gpt3_gpipe_b64_1_l2048_m768_w12_p3_d4
# gpt3_gpipe_b64_1_l2048_m768_w48_p3_d16
# gpt3_gpipe_b64_1_l2048_m2048_w12_p12_d1
# gpt3_gpipe_b64_1_l2048_m2048_w48_p12_d4

# gpt3_gpipe_allreduce_b64_1_l2048_m768_w3_p3_d1

if [ $# -eq 1 ]
then
  profix=$1

  postfixes=(
    "tidy_profiling_default"
    #"fp16_offload_tidy_profiling_default"
    #"fp16_offload_tidy_profiling_b1"
    #"fp16_offload_tidy_profiling_d1b5"
    #"fp16_offload_tidy_profiling_d5b2"
    #"fp16_offload_tidy_profiling_d10b1"
    #"fp16_offload_tidy_profiling_d50b1"
    #"fp16_offload_tidy_profiling_heter6"
  )

  world_size=${#ips[@]}

  if [ -d "../trace_json/${profix}" ]
  then
    rm -rf ../trace_json/"$profix"
  fi

  mkdir ../trace_json/"$profix"

  for postfix in "${postfixes[@]}"
  do
    for rank in "${!ips[@]}"
    do
      scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}":~/GPT-home-private/trace_json/"$profix"_"$rank"_"$postfix".json ../trace_json/"$profix" &
    done
  done
  wait
fi

if [ $# -eq 2 ]
then
  profix=$1
  postfix=$2

  for rank in "${!ips[@]}"
  do
    scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}":~/GPT-home-private/trace_json/"$profix"_"$rank"_"$postfix".json ../trace_json/"$profix" &
  done
  wait
fi
