source ./ip_list.sh

scp fsuser@"${ips[0]}":"/home/fsuser/GPT-home-private/logs/*rank0*.log" ../logs