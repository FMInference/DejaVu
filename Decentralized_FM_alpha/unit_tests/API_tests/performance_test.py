import time
import requests
import argparse


def generate_post_dict(model_name: str, input_file: str, prompt_num: int, max_tokens: int):
    plannet_post_dict = {
        "type": "general",
        "payload": {
            "max_tokens": max_tokens,
            "n": 1,
            "temperature": 0.8,
            "top_p": 0.6,
            "top_k": 5,
            "model": model_name,
            "prompt": [],
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
    with open(input_file, 'r') as fp:
        line = fp.readline()
        for i in range(prompt_num):
            # Get next line from file
            plannet_post_dict['payload']["prompt"].append(line)
    return plannet_post_dict


def main():
    parser = argparse.ArgumentParser(description='Load and Save.')
    parser.add_argument('--prompt-num', type=int, default=1, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--prompt-tokens', type=int, default=128, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--max-tokens', type=int, default=32, metavar='N',
                        help='-')
    parser.add_argument('--model-name', type=str, default='glm', metavar='S',
                        help='-')
    args = parser.parse_args()
    plannet_post_dict = generate_post_dict(model_name=args.model_name, input_file=f'foo_txt_{args.prompt_tokens}.txt',
                                           prompt_num=args.prompt_num, max_tokens=args.max_tokens)

    post_res = requests.post("https://planetd.shift.ml/jobs", json=plannet_post_dict).json()
    print(post_res['id'])
    get_res = None
    while True:
        time.sleep(3)
        get_res = requests.get("https://planetd.shift.ml/job/{}".format(post_res['id'])).json()
        print(get_res['status'])
        if get_res['status'] == 'finished':
            returned_payload = get_res['returned_payload']
            # print(returned_payload)
            if args.prompt_num == 1:
                print(f"Model: {args.model_name}, input_token: {args.prompt_tokens}, output_token: {args.max_tokens} "
                      f"latency: {returned_payload['raw_compute_time']}")
            else:
                print(f"Model: {args.model_name}, input_token: {args.prompt_tokens}, output_token: {args.max_tokens} "
                      f"latency: {returned_payload['raw_compute_time']} s, "
                      f"throughput: {args.max_tokens*args.prompt_num/returned_payload['raw_compute_time']} token/s")
            break


if __name__ == '__main__':
    main()

