server_ip=$1
job_name=$2
submit_count=$3
interval=$4

for((i=0; i<${submit_count}; i++))
do
  sleep $interval
  python job_submit_client.py --coordinator-server-ip $server_ip --submit-job inference --job-name $job_name
done
