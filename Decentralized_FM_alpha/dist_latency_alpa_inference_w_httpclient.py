from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
from time import sleep
import argparse
import time
import math
import torch
import random
import numpy as np
from alpa.device_mesh import set_seed


def get_tokenizer_model(args):
    if args.model_name == 'opt-175b':
        # The 30B version works for all OPT models.
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/opt-175b", path="/root/fm/models/alpa_models/")

    elif args.model_name == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom')
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/bloom", path="/root/fm/models/alpa_models/")

    elif args.model_name == 'bloomz':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloomz')
        tokenizer.add_bos_token = False
        # llm_serving does not recoginze bloomz, since the model parameter is from bloomz,
        # this should be fine
        model = get_model(model_name="alpa/bloom", path="/root/fm/models/alpa_models/bloomz-np")
    else:
        assert False, f"Not legal name {args.model_name}"

    return tokenizer, model


def post_processing_text(input_text, output_text, query):
    print(f"<post_processing_text> input_text: {input_text}")
    print(f"<post_processing_text> output_text: {output_text}")
    stop_tokens = []
    if query.get('stop', []) is not None:
        for token in query.get('stop', []):
            if token != '':
                stop_tokens.append(token)
    print(f"<post_processing_text> stop_tokens: {stop_tokens}.")

    if query.get('max_tokens') == 0:
        return ""

    if not query.get('echo', False):
        text = output_text[len(input_text):]
    else:
        text = output_text
    end_pos = len(text)
    print(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in stop_tokens:
        if query.get('echo', False):
            if text[len(input_text):].find(stop_token) != -1:
                end_pos = min(text[len(input_text):].find(stop_token) + len(stop_token), end_pos)
        else:
            if text.find(stop_token) != -1:
                end_pos = min(text.find(stop_token) + len(stop_token), end_pos)
        print(f"<post_processing_text>2 end_pos: {end_pos}.")

    print(f"<post_processing_text> text: {text}, end_pos: {end_pos}")
    post_processed_text = text[:end_pos + 1]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def to_result(input_text, output_text, query):
    result = {}
    items = []
    for i in range(len(output_text)):
        item = {'choices': [], }
        print(f"<to_result> output{i}: {len(input_text[i])} / {len(output_text[i])}")
        choice = {
            "text": post_processing_text(input_text[i], output_text[i], query),
            "index": 0,
            "finish_reason": "length"
        }
        item['choices'].append(choice)
        items.append(item)
    result['inference_result'] = items
    return result


def main():
    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--job-id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/root/fm/working_dir', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--model-name', type=str, default='bloom', metavar='S',
                        help='trained model path')
    parser.add_argument('--batch-size', type=int, default=1, metavar='S',
                        help='batch-size for inference (default:8)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    args = parser.parse_args()
    print(args)

    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )
    try:
        tokenizer, model = get_tokenizer_model(args)
        local_cord_client.update_status(args.job_id, "running")
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    print(f"{args.model_name} Initialized.")

    try:
        while True:
            job_id = None
            raw_text = None
            try:
                instructions = local_cord_client.fetch_instructions(args.model_name, 0)
                last_instruction = instructions[-1]

                if last_instruction["message"] == "break":
                    logger.info("Received stop instruction.")
                    logger.info("# BREAK ")
                    break
                elif last_instruction["message"] == "continue":
                    logger.info(f"Received keep instruction. <{args.model_name}>")
                    sleep(1)
                elif last_instruction["message"] == "run":
                    fetched_tasks = [x for x in instructions
                                     if x["message"] == "run" and x['payload']['status'] == 'submitted']

                    if len(fetched_tasks) > 0:
                        instruction = fetched_tasks[0]
                        logger.info("Instruction:")
                        logger.info(str(instruction))
                        # TODO: we assume len(payload) is 1, right?
                        query = instruction['payload']['payload'][0]
                        if isinstance(query['prompt'], list):
                            raw_text = query['prompt']
                        elif isinstance(query['prompt'], str):
                            raw_text = [query['prompt']]
                        else:
                            print("wrong prompt format, it can only be str or list of str")
                            print(query['prompt'])

                        job_id = instruction['payload']['id']
                        print(f"Job <{job_id}> has been processed")

                        start_time = time.time()

                        batch_size = 1
                        num_iter = math.ceil(len(raw_text) / batch_size)
                        answers = []
                        seed = query.get('seed', None)
                        if seed is not None:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            random.seed(seed)
                            set_seed(seed)

                        for iter_i in range(num_iter):
                            current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
                            input_ids = tokenizer(current_raw_text, return_tensors="pt").input_ids
                            if query.get('temperature', 0.9) == 0:
                                output = model.generate(input_ids=input_ids,
                                                        max_new_tokens=query.get('max_tokens', 16),
                                                        temperature=1.0,
                                                        top_k=1,
                                                        top_p=query.get('top_p', 0),
                                                        do_sample=True)
                            else:
                                output = model.generate(input_ids=input_ids,
                                                        max_new_tokens=query.get('max_tokens', 16),
                                                        temperature=query.get('temperature', 0.9),
                                                        top_k=query.get('top_k', 50),
                                                        top_p=query.get('top_p', 0),
                                                        do_sample=True)
                            generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)
                            answers.extend(generated_string)

                        end_time = time.time()
                        print(f"Job-{job_id} {args.model_name} Inference takes {end_time - start_time}s")
                        # print(f"outputs by hf model: {outputs}")
                        result = to_result(raw_text, answers, query)
                        return_payload = {
                            'request': query,
                            'result': result,
                            'raw_compute_time': end_time - start_time
                        }
                        # local_cord_client.update_status(
                        local_cord_client.update_status_global_coordinator(
                            job_id,
                            "finished",
                            returned_payload=return_payload
                        )
                        local_cord_client.update_status(job_id, "finished", returned_payload={})

            except Exception as e:
                error = traceback.format_exc()
                local_cord_client.update_status(
                    job_id,
                    "failed",
                    returned_payload={"message": error}
                )
                print(error)

    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == "__main__":
    main()
