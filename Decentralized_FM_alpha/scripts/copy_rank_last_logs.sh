if [ $# -eq 1 ]
  then
  ip=$1
  scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip":"/home/ubuntu/GPT-home-private/logs/*.log" ../logs
else
  source ./ip_list.sh
  scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[-1]}":"/home/ubuntu/GPT-home-private/logs/*.log" ../logs
fi