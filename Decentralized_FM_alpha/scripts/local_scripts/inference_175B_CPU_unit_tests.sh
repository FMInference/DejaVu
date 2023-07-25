cd ~/GPT-home-private

source activate pytorch_p38



batches=(2 4 8 16 32 48 64)

for batch in "${batches[@]}"
do
  echo "Batch size ${batch}"
  python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96 --batch-size $batch >> ./logs/local_175B_cpu_inference_b"${batch}".log
done