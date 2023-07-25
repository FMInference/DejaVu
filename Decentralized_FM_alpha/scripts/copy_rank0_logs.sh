source ./ip_list.sh

scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[0]}":"/home/ubuntu/GPT-home-private/logs/*.log" ../logs