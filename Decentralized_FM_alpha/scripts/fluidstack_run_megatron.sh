source ./ip_list.sh

world_size=${#ips[@]}


script=$1

PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3

num_layers=$4
global_batch_size=$5

for rank in "${!ips[@]}"
do
  ip=${ips[rank]}
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  ssh fsuser@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$world_size" "$rank" "$num_layers" "$global_batch_size"&

done
wait