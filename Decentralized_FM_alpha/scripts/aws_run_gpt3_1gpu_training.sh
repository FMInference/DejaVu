source ./ip_list.sh

world_size=${#ips[@]}

script=$1

ga_step=$2
num_layers=$3
batch_size=$4

log_mode='optimal_map'

for rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $# -eq 4 ]
  then
    echo "Running in default network."
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode"&
  elif [ $# -eq 5 ]
  then
    case=$5
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$case" &
  elif [ $# -eq 6 ]
  then
    delay_ms=$5
    rate_gbit=$6
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done
wait