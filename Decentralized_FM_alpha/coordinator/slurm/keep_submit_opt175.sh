i=0
while :
do

  file=/sailhome/biyuan/fm/queries_opt_175b/request_${i}.jsonl
  outfile=/sailhome/biyuan/fm/queries_opt_175b/output_request_${i}.jsonl
  if [ -f "$file" ]; then
  if [ ! -f "$outfile" ]; then
      python fill_template.py --infer-data $file --template-path slurm_scripts/slurm_opt175_inf_template.sh --output-path slurm_scripts/_slurm_inf.sh
      python job_submit_client.py --coordinator-server-ip 10.79.12.70  --submit-job inference --job-name "_slurm_inf"
      sleep 600
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
