import requests


plannet_post_dict = {
    "type": "general",
    "payload": {
        "max_tokens": 16,
        "n": 1,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 5,
        "model": "bloomz",
        "prompt": ["Where is Zurich?"],
        "request_type": "language-model-inference",
        "stop": [],
        "best_of": 1,
        "logprobs": 0,
        "echo": False,
        "prompt_embedding": False
    },
    "returned_payload": {},
    "status": "submitted",
    "source": "dalle",
}


together_post_dict = {
    "source": "toma_web",
    "prompt": [
        "where is Zurich?",
        "Where is LA's airport?",
        "Where is Houston?",
        "Where is Austin's train station?"
    ],
    "model": "gptj6b",
    "model_owner": "together",
    "owner": "binhang",
    "tags": [
        "string"
    ],
    "num_returns": 1,
    "args": {},
    "request_type": "language-model-inference",
    "seed": 0
}

res = requests.post("https://planetd.shift.ml/jobs", json=plannet_post_dict).json()

#res = requests.put("https://zentrum.xzyao.dev/jobs/jobs", json=together_post_dict).json()

print(res)
