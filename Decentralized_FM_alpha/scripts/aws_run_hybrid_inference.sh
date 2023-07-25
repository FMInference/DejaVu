source ./ip_list.sh

world_size=${#ips[@]}

script=$1

num_gpus=$2


for rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $rank -lt $num_gpus ]
  then
    echo "Issue cmd on GPU node."
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$num_gpus" "$rank" "GPU"
  else
    echo "Issue cmd on CPU node."
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$num_gpus" "$rank" "CPU"
  fi &
done
wait