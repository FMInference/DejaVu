source ./ip_list.sh

world_size=${#ips[@]}

rank_map=(0 2 10 4 12 9 14 13 3 7 8 6 1 11 5 15)

script=$1

PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3

num_layers=$4
global_batch_size=$5

for i in "${!ips[@]}"
do
  rank=${rank_map[$i]}
  ip=${ips[$i]}
  echo "Issue command $script in Rank-$rank node: ${ip}"
  ssh fsuser@"${ip}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$world_size" "$rank" "$num_layers" "$global_batch_size"&
done
wait