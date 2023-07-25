source ./ip_list.sh

nodes_per_node=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
world_size=32

script=$1
num_layer=$2
ga_step=$3


# Random seed 2022
# rank_map=(0 27 16 6 12 9 8 7 25 21 1 28 18 22 3 14 20 26 17 11 2 23 24 15 13 29 31 10 19 5 30 4) # for complete shuffle
rank_map=(0 2 10 4 12 9 14 13 3 7 8 6 1 11 5 15)

for index in "${!ips[@]}"
do
  node_rank=${rank_map[$index]}
  ip=${ips[$index]}
  echo "Issue command $script in Rank-${node_rank} node: $ip"
  for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
  do
    let "global_rank = $node_rank*2+$i"
    echo "Node rank:$global_rank, local rank: $i"
    ssh fsuser@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$global_rank" "$i" "$num_layer" "$ga_step" "wo_scheduler" &
  done
done
wait
