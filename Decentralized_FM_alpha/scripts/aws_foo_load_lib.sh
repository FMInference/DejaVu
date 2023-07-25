source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/foo_load_lib.sh &
done
wait