source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  ssh fsuser@"$ip" "bash -s" < ./local_scripts/local_kill_process.sh &
done
wait