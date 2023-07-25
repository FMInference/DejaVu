source ./ip_list.sh

world_size=${#ips[@]}

script=$1

PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3
GPUS_PER_NODE=$4

num_layers=$5
global_batch_size=$6

for rank in "${!ips[@]}"
do
  ip=${ips[rank]}
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $rank -eq 63 ]
  then
    echo "========Last rank IP ${ip}==========="
  fi
  if [ $# -eq 6 ]
  then
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank" "$num_layers" "$global_batch_size"&
  elif [ $# -eq 7 ]
  then
    case=$7
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank"  "$num_layers" "$global_batch_size" "$case" &
  elif [ $# -eq 8 ]
  then
    delay_ms=$7
    rate_gbit=$8
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank"  "$num_layers" "$global_batch_size" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done
wait