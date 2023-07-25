case=$1

bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 2 3 64 $case
sleep 5s
bash copy_rank0_logs.sh

#bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 4 3 64 $case
#sleep 5s
#bash copy_rank0_logs.sh

#bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 8 3 64 $case
#sleep 5s
#bash copy_rank0_logs.sh

#bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 2 4 64 $case
#sleep 5s
#bash copy_rank0_logs.sh

bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 4 4 64 $case
sleep 5s
bash copy_rank0_logs.sh

#bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 8 4 64 $case
#sleep 5s
#bash copy_rank0_logs.sh

#bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 2 5 62 $case
#sleep 5s
#bash copy_rank0_logs.sh

#bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 4 5 62 $case
#sleep 5s
#bash copy_rank0_logs.sh

bash aws_run_gpt3_1gpu_training.sh gpt3_xl_pp8_dp8.sh 8 5 62 $case
sleep 5s
bash copy_rank0_logs.sh