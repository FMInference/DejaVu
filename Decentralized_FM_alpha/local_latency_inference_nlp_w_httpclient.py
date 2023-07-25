import argparse
import time
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
from time import sleep
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import math
import numpy as np
import random
import torch

from transformers import GPTJForCausalLM,GPTNeoXForCausalLM
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_utils import no_init_weights
import os


def create_emtpy_gptj(config):
    import torch.nn as nn
    _reset_parameters_linear = nn.Linear.reset_parameters

    def dummy(*args, **kargs):
        pass

    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = GPTJForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model


def create_emtpy_gptneox(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = GPTNeoXForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint_gpt_j_6b(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.transformer.h)
    assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]: v for k, v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.transformer.wte.weight.data[:] = _tmp['wte.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j + 1}."):]: v for k, v in checkpoint.items() if k.startswith(f"{j + 1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.transformer.h[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]: v for k, v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.transformer.h[i * n_layer_per_stage + j].load_state_dict(_tmp)

            _tmp = {k[len(f"{n_layer_per_stage}."):]: v
                    for k, v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.transformer.ln_f.weight.data[:] = _tmp['ln_f.weight']
            model.transformer.ln_f.bias.data[:] = _tmp['ln_f.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]: v for k, v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.transformer.h[i * n_layer_per_stage + j].load_state_dict(_tmp)

    return model


def load_decentralized_checkpoint_gpt_neox(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.gpt_neox.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.gpt_neox.embed_in.weight.data[:] = _tmp['embed_in.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.gpt_neox.layers[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                if i*n_layer_per_stage + j == 44:
                    break
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.gpt_neox.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

            _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.gpt_neox.final_layer_norm.weight.data[:] = _tmp['final_layer_norm.weight']
            model.gpt_neox.final_layer_norm.bias.data[:] = _tmp['final_layer_norm.bias']
            model.embed_out.weight.data[:] = _tmp['embed_out.weight']
            if 'lm_head.bias' in _tmp:
                model.embed_out.bias.data[:] = _tmp['embed_out.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.gpt_neox.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


def get_huggingface_tokenizer_model(args, device):
    if args.model_name == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16)
    elif args.model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b', model_max_length=512)
        # tokenizer.model_max_length=512
        model = T5ForConditionalGeneration.from_pretrained('t5-11b', torch_dtype=torch.bfloat16)
        model.config.eos_token_id = None
    elif args.model_name == 't0pp':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/T0pp')
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", torch_dtype=torch.bfloat16)
    elif args.model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/ul2')
        model = T5ForConditionalGeneration.from_pretrained("google/ul2", torch_dtype=torch.bfloat16)
    elif args.model_name == 'gpt-j-6b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
    elif args.model_name == 'Together-gpt-JT-6B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float16)
    elif args.model_name == 'gpt-neox-20b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)
    elif args.model_name == 'Together-gpt-J-6B-ProxAdam-50x':
        config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        model = create_emtpy_gptj(config).half().eval()
        load_decentralized_checkpoint_gpt_j_6b(model, '/root/fm/models/Together-gpt-J-6B-ProxAdam-50x',
                                               n_stages=2, n_layer_per_stage=14)
    elif args.model_name == 'Together-gpt-neox-20B':
        config = AutoConfig.from_pretrained('EleutherAI/gpt-neox-20b')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        model = create_emtpy_gptneox(config).half().eval()
        load_decentralized_checkpoint_gpt_neox(model, '/root/fm/models/Together-gpt-neox-20B', n_stages=8,
                                               n_layer_per_stage=6)
    else:
        assert False, "Model not supported yet."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    model = model.to(device)
    return tokenizer, model


def pre_processing_texts(input_text, model_name, tokenizer):
    ed_input_text = []
    for i in range(len(input_text)):
        current_tokens = tokenizer(input_text[i], padding=False, truncation=False, return_tensors="pt")
        current_output = tokenizer.decode(current_tokens['input_ids'][0]).replace("</s>", "")
        ed_input_text.append(current_output)
    if model_name == 't5-11b' or model_name == 'ul2':
        for i in range(len(ed_input_text)):
            ed_input_text[i] = ed_input_text[i] + "<extra_id_0>"
            input_text[i] = input_text[i] + "<extra_id_0>"
    print(f"<pre_processing_texts> input_text: {input_text}, ed_input_text: {ed_input_text}")
    return input_text, ed_input_text


def model_to_max_token(model_name, query):
    if model_name == 't5-11b' or model_name == 'ul2':
        return 1 + query.get('max_tokens', 16)
    # elif model_name == 't0pp':
    #    return 1 + query.get('max_tokens', 16)
    else:
        return query.get('max_tokens', 16)


def post_processing_text(input_text, output_text, model_name, query):
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

    if model_name == 'gpt-j-6b' or model_name == 'gpt-neox-20b' or model_name == 'Together-gpt-JT-6B-v1' \
            or model_name == 'Together-gpt-J-6B-ProxAdam-50x' or model_name == 'Together-gpt-neox-20B':
        if not query.get('echo', False):
            text = output_text[len(input_text):]
        else:
            text = output_text
        end_pos = len(text)
        print(f"<post_processing_text>1 end_pos: {end_pos}.")
        for stop_token in stop_tokens:
            if query.get('echo', False):
                if text[len(input_text):].find(stop_token) != -1:
                    end_pos = min(text[len(input_text):].find(stop_token), end_pos)
            else:
                if text.find(stop_token) != -1:
                    end_pos = min(text.find(stop_token), end_pos)
            print(f"<post_processing_text>2 end_pos: {end_pos}.")
    elif model_name == 'ul2' or model_name == 't0pp' or model_name == 't5-11b' or model_name == 'flan-t5-xxl':
        if model_name == 't5-11b' or model_name == 'ul2':
            input_text = input_text.replace("<extra_id_0>", "")
        if query.get('echo', False):
            text = input_text + ' ' + output_text
        else:
            text = output_text
        end_pos = len(text)
        print(f"<post_processing_text>1 end_pos: {end_pos}.")
        for stop_token in stop_tokens:
            if query.get('echo', False):
                if text[len(input_text) + 1:].find(stop_token) != -1:
                    end_pos = min(text[len(input_text) + 1:].find(stop_token) + len(stop_token), end_pos)
            else:
                if text.find(stop_token) != -1:
                    end_pos = min(text.find(stop_token), end_pos)
            print(f"<post_processing_text>2 end_pos: {end_pos}.")
    else:
        assert False, "Model not supported yet."
    print(f"<post_processing_text> text: {text}, end_pos: {end_pos}")
    post_processed_text = text[:end_pos]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def to_result(input_text, output_text, model_name, query):
    result = {}
    items = []
    for i in range(len(output_text)):
        item = {'choices': [], }
        print(f"<to_result> output{i}: {len(input_text[i])} / {len(output_text[i])}")
        choice = {
            "text": post_processing_text(input_text[i], output_text[i], model_name, query),
            "index": 0,
            "finish_reason": "length"
        }
        item['choices'].append(choice)
        items.append(item)
    result['inference_result'] = items
    return result


def main():
    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--model-name', type=str, default='t5-11b', metavar='S',
                        help='trained model path')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='S',
                        help='cuda-id (default:0)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='S',
                        help='batch-size for inference (default:8)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    args = parser.parse_args()
    print(args)
    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )
    assert (torch.cuda.is_available())
    device = torch.device('cuda', args.cuda_id)
    try:
        tokenizer, model = get_huggingface_tokenizer_model(args, device)
        local_cord_client.update_status(args.job_id, "running")
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

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

                        raw_text, ed_raw_text = pre_processing_texts(raw_text, args.model_name, tokenizer)

                        print(f"<main> input_text: {raw_text}, ed_input_text: {ed_raw_text}")

                        batch_size = min(len(raw_text), args.batch_size)
                        num_iter = math.ceil(len(raw_text) / batch_size)
                        answers = []
                        seed = query.get('seed', None)
                        if seed is not None:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            random.seed(seed)

                        for iter_i in range(num_iter):
                            current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
                            inputs = tokenizer(
                                current_raw_text,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                            )
                            inputs.to(device)
                            if query.get('temperature', 0.9) == 0:
                                outputs = model.generate(
                                    **inputs, do_sample=True, top_p=query.get('top_p', 0),
                                    temperature=1.0, top_k=1,
                                    max_new_tokens=model_to_max_token(args.model_name, query),
                                    return_dict_in_generate=True,
                                    output_scores=True,  # return logit score
                                    output_hidden_states=True,  # return embeddings
                                )
                            else:
                                outputs = model.generate(
                                    **inputs, do_sample=True, top_p=query.get('top_p', 0),
                                    temperature=query.get('temperature', 0.9),
                                    max_new_tokens=model_to_max_token(args.model_name, query),
                                    return_dict_in_generate=True,
                                    output_scores=True,  # return logit score
                                    output_hidden_states=True,  # return embeddings
                                )

                            current_output_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                            print(f"<Include_special_tokens>:", current_output_texts)
                            current_output_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                            answers.extend(current_output_texts)

                        end_time = time.time()
                        print(f"Job-{job_id} {args.model_name} Inference takes {end_time - start_time}s")
                        # print(f"outputs by hf model: {outputs}")
                        result = to_result(ed_raw_text, answers, args.model_name, query)
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
