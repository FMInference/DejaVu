source ./ip_list.sh



for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  if [ $# -eq 0 ]
  then
    ssh fsuser@"$ip" "bash -s" < ./local_scripts/local_git_pull.sh &
  else
    token=$1
    ssh fsuser@"$ip" "bash -s" < ./local_scripts/local_git_pull.sh "$token" &
  fi
done
wait