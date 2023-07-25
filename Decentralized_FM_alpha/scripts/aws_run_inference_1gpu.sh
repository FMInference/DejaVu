source ./ip_list.sh

world_size=${#ips[@]}

script=$1

for rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  echo "Running in default network."
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" &
done
wait