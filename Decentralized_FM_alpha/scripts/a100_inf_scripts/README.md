# FM Inference

## Preparation

- Python dependencies:
  - pytorch >= 1.10.0
  - transformers==4.21.1 **// this is important for Bloom**
  - sentencepiece **// this is needed for models depending on sentencepiece tokenizers**
  - flask **// for lantency-oriented server**
- Acquire model checkpoints. Please refer to [here](https://docs.google.com/spreadsheets/d/1cXzqSH6qkaydhb4zecs4aO3bWFU8RdH9KYuwexelzqM/edit?usp=sharing).

Note: Sometimes, the checkpoint may miss some model/tokenizer config files. If this happens, please tell me (@Jue) or add them manually with the following code:

```python
from transformers import AutoConfig, AutoTokenizer

model_name_on_huggingface = 'bigscience/bloom' # change to the model name on huggingface
model_path_to_save = './models/bloom-new/'

AutoConfig.from_pretrained(model_name_on_huggingface).save_pretrained(model_path_to_save)
AutoTokenizer.from_pretrained(model_name_on_huggingface).save_pretrained(model_path_to_save)
```

## Data Format

We generally follow OpenAI's API (https://beta.openai.com/docs/api-reference/completions). 

Each line of the data file is a json object. The below is an example:

```json
{
    "best_of":1,
    "echo":false,
    "engine":"gpt-j-6b",
    "logprobs":1,
    "max_tokens":2,
    "n":1,
    "prompt":"Passage: The bedroom is north of the garden.\nThe bathroom is west of the garden.\nThe office is east of the garden.\nThe hallway is west of the bathroom.\nThe bathroom is south of the kitchen.\nQuestion: How do you go from the bathroom to the office?\nAnswer: east east\n\nPassage: The office is north of the hallway.\nThe bathroom is east of the bedroom.\nThe hallway is north of the bedroom.\nThe office is east of the kitchen.\nThe bedroom is north of the garden.\nQuestion: How do you go from the bedroom to the office?\nAnswer: north north\n\nPassage: The garden is south of the bedroom.\nThe bathroom is east of the kitchen.\nThe office is north of the bathroom.\nThe bathroom is north of the hallway.\nThe bedroom is west of the office.\nQuestion: How do you go from the bathroom to the bedroom?\nAnswer: north west\n\nPassage: The garden is west of the office.\nThe bedroom is west of the garden.\nThe bathroom is south of the office.\nThe office is west of the hallway.\nThe kitchen is north of the office.\nQuestion: How do you go from the kitchen to the garden?\nAnswer: south west\n\nPassage: The bedroom is west of the office.\nThe garden is south of the bedroom.\nThe bathroom is south of the kitchen.\nThe hallway is north of the bedroom.\nThe bedroom is east of the bathroom.\nQuestion: How do you go from the bathroom to the hallway?\nAnswer: east north\n\nPassage: The bedroom is north of the bathroom.\nThe bedroom is east of the office.\nThe kitchen is east of the bedroom.\nThe garden is west of the bathroom.\nThe hallway is south of the bathroom.\nQuestion: How do you go from the bathroom to the office?\nAnswer:",
    "stop":[
        "\n"
    ],
    "temperature":0.0,
    "top_p":1,
}
```

Each line of the output file is also a json object. The below is an example:

```json
{
    "request":{
        "best_of":1,
        "echo":false,
        "engine":"gpt-j-6b",
        "logprobs":1,
        "max_tokens":2,
        "n":1,
        "prompt":"Passage: The bedroom is north of the garden.\nThe bathroom is west of the garden.\nThe office is east of the garden.\nThe hallway is west of the bathroom.\nThe bathroom is south of the kitchen.\nQuestion: How do you go from the bathroom to the office?\nAnswer: east east\n\nPassage: The office is north of the hallway.\nThe bathroom is east of the bedroom.\nThe hallway is north of the bedroom.\nThe office is east of the kitchen.\nThe bedroom is north of the garden.\nQuestion: How do you go from the bedroom to the office?\nAnswer: north north\n\nPassage: The garden is south of the bedroom.\nThe bathroom is east of the kitchen.\nThe office is north of the bathroom.\nThe bathroom is north of the hallway.\nThe bedroom is west of the office.\nQuestion: How do you go from the bathroom to the bedroom?\nAnswer: north west\n\nPassage: The garden is west of the office.\nThe bedroom is west of the garden.\nThe bathroom is south of the office.\nThe office is west of the hallway.\nThe kitchen is north of the office.\nQuestion: How do you go from the kitchen to the garden?\nAnswer: south west\n\nPassage: The bedroom is west of the office.\nThe garden is south of the bedroom.\nThe bathroom is south of the kitchen.\nThe hallway is north of the bedroom.\nThe bedroom is east of the bathroom.\nQuestion: How do you go from the bathroom to the hallway?\nAnswer: east north\n\nPassage: The bedroom is north of the bathroom.\nThe bedroom is east of the office.\nThe kitchen is east of the bedroom.\nThe garden is west of the bathroom.\nThe hallway is south of the bathroom.\nQuestion: How do you go from the bathroom to the office?\nAnswer:",
        "stop":[
            "\n"
        ],
        "temperature":0.0,
        "top_p":1,
    },
    "result":{
        "choices":[
            {
                "text":" north east",
                "index":0,
                "logprobs":{
                    "tokens":[
                        "\u0120north",
                        "\u0120east"
                    ],
                    "token_logprobs":[
                        -0.9970703125,
                        -0.54150390625
                    ],
                    "top_logprobs":[
                        {
                            "\u0120north":-0.9970703125
                        },
                        {
                            "\u0120east":-0.54150390625
                        }
                    ],
                    "text_offset":[
                        
                    ]
                },
                "finish_reason":"length"
            }
        ],
        "request_time":{
            "batch_time":2.202444076538086,
            "batch_size":29
        }
    }
}
```

Currently, we need to keep hyperparameters (temperature, top\_p, stop, etc.) the same during inference. So we shall split the data manually. We can use `split_queries.py` in this folder.

## Launch local runs

Take OPT-175b as an example:

```bash
i=0
while :
do

  file=/data/queries_opt175/request_${i}.jsonl
  outfile=/data/queries_opt175/output_request_${i}.jsonl
  if [ -f "$file" ]; then
  if [ ! -f "$outfile" ]; then
      
      echo "start running ${file}"
  
      ARGS="--model-name /data/models/opt-175b-new \
      --model-type opt \
      --seed 42 \
      --fp16 \
      --num-layers 12 \
      --max-layers 96 \
      --budget 26800 \
      --num-iters 99999999999999 \
      --dist-url tcp://127.0.0.1:9031 \
      --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
      --pp-mode pipe_sync_sample_mask_token_pipe \
      --infer-data ${file}"

      (trap 'kill 0' SIGINT; \
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
          &
      python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
          & \
      wait)
  fi
  fi

  ((i++))
  # Max id of jobs
  if [ "$i" -ge "500" ]; then
    echo "stopping submitting tasks"
    break
  fi
  sleep 0.01
done
```

- When `--infer-data` is given, hyperparameters will be automatically set. `--budget` needs to be specified, and it should be the maximum tokens in a batch (e.g. max batch size 32 for a sequence length of 512, the budget should be 32\*512=16384).
- When `--infer-data` is not given, dummy data will be used. In such case, please specify all hyperparameters.
- `--max-layers` will limit the max layers globally. E.g. pipe size == 8, but bloom has 70 layers, we shall set `--num-layers 9 --max-layers 70`, where 8\*9=72 but the last two layers will be dropped.
- `--num-iters` should be an arbitrary large number.



## Launch runs with coordinator

Everything should be trivial. But there is a hack when submitting the job: the infer data path needs to be appended after the lsf script name `--job-name "lsf_optxxx#${infer_data_path}"`.
