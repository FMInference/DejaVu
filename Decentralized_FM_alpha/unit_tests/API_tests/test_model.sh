model_name=$1
prompt_nums=(1 16)
if [ $model_name == "glm" ]
then
  prompt_tokens=(128 512 2048)
else
  prompt_tokens=(128 512)
fi
generate_tokens=(32 64 128)

for prompt_num in "${prompt_nums[@]}"
do
  for prompt_token in "${prompt_tokens[@]}"
  do
    for generate_token in "${generate_tokens[@]}"
    do
      python3 performance_test.py --model-name $model_name --prompt-num $prompt_num --prompt-tokens $prompt_token --max-tokens $generate_token
    done
  done
done