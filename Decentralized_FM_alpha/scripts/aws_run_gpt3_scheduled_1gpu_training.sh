source ./ip_list.sh

world_size=${#ips[@]}

script=$1

# This is need to training scheduled examples. Use generate_assignment.py to generate them (copy the result printout here).
# Random seed 2022
# rank_map=(0 2 32 33 4 10 7 45 36 8 51 26 11 5 53 1 40 23 37 14 13 43 54 21 57 35 63 18 6 24 16 22 38 3 58 61 44 27 52 30 15 9 39 47 48 41 31 20 12 28 34 42 17 55 19 25 56 60 59 50 49 46 29 62)
# Random seed 2023
rank_map=(0 42 36 19 15 13 55 11 1 10 43 45 39 12 63 5 37 59 35 31 20 27 17 28 41 3 62 21 47 32 22 51 46 2 9 44 16 61 30 52 8 50 58 57 25 6 40 14 49 48 18 54 33 38 23 60 53 4 29 34 56 7 26 24)
# Random seed 2024
# rank_map=(0 35 24 52 16 49 41 4 18 7 45 13 22 60 14 15 51 25 17 8 47 55 19 63 21 57 44 26 5 58 20 50 30 6 54 43 23 34 46 27 39 10 40 62 29 12 32 53 48 31 38 61 3 11 36 2 42 56 37 28 1 33 59 9)

# 4_2 scheduled
#rank_map=(63 47 15 31 50 55 28 20 25 1 26 30 61 51 52 38 34 35 13 11 27 36 41 37 14 43 12 23 7 3 49 33 10 54 0 29 60 8 22 58 16 21 17 18 44 24 4 2 6 32 46 59 53 5 39 9 40 62 57 45 48 19 42 56)

ga_step=$2
num_layers=$3
batch_size=$4

log_mode=$5

for i in "${!ips[@]}"
do
  rank=${rank_map[$i]}
  ip=${ips[$i]}
  echo "Issue command $script in Rank-$rank node: ${ips[$i]}"
  if [ $rank -eq 0 ]
  then
    echo "Rank 0 on $i-th instances."
  fi
  if [ $# -eq 5 ]
  then
    echo "Running in default network."
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" &
  elif [ $# -eq 6 ]
  then
    case=$6
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$case" &
  elif [ $# -eq 7 ]
  then
    delay_ms=$6
    rate_gbit=$7
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done
wait