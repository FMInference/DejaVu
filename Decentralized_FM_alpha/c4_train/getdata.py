from datasets import load_dataset
from tqdm import tqdm
import json


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


dataset = load_dataset("c4", "en", split="train", streaming=True)
dataset = dataset.shuffle(buffer_size=10000, seed=42)
path = "c4_train.jsonl"

for idx, doc in enumerate(tqdm(dataset)):
    data = {
        "best_of": 1,
        "echo": True,
        "logprobs": 1,
        "max_tokens": 0,
        "model": "opt-175b",
        "n": 1,
        "prompt": doc["text"],
        "request_type": "language-model-inference",
        "stop": None,
        "temperature": 0,
        "top_p": 1,
    }
    dump_jsonl([data], path, append=True)
    if idx == 500:
        print("Collected 500 samples")
        break
