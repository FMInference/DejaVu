N_QUEUE=300
i=3
while :
do
  while :
  do
  n=`bjobs | wc -l`
  n=`echo "print(max($n-1, 0))" | python`
  echo "queue: ${n}/${N_QUEUE}"
  if [ "$n" -le "${N_QUEUE}" ]; then
    break
  fi
  echo "queue full, sleep 180s"
  sleep 180
  done

  file=/cluster/home/juewang/fm/queries_t5_11b/request_${i}.jsonl
  outfile=/cluster/home/juewang/fm/queries_t5_11b/output_request_${i}.jsonl
  if [ -f "$file" ]; then
  if [ ! -f "$outfile" ]; then
      python job_submit_client.py --coordinator-server-ip 129.132.93.89 --submit-job inference --job-name "lsf_t5#$file"
      sleep 10
  fi
  fi

  ((i++))
  # Max id of jobs
  if [ "$i" -ge "500" ]; then
    echo "stopping submitting tasks"
    break
  fi
  sleep 0.1
done
