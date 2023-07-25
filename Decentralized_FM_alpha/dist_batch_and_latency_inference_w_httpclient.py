import os
import time
from time import sleep
import argparse
from loguru import logger
from pipeline_parallel.dist_pp_utils import get_pp_inference_module
from utils.dist_args_utils import *
from utils.dist_inference_utils import *
from comm.comm_utils import *
from coordinator.http_coordinate_client import get_coordinator_client, init_coordinator_client, alias_to_model_name
from coordinator.coordinator_client import LocalCoordinatorClient # TODO: merge two coor clients
from task_datasets.inference_data import get_request_processor
import traceback


def update_setting(args, pipeline, query):
    
    # update pipline
    pipeline.echo_prompt = query.get('echo', False)
    pipeline.top_k_per_token = query.get('logprobs', 0)
    pipeline.generate_seq_length = query.get('max_tokens', 1)
    pipeline.num_completions = query.get('n', 1)
    pipeline.stop = query.get('stop', None)
    pipeline.temperature = query.get('temperature', 0)
    pipeline.top_p = query.get('top_p', 1.0)

    # in latency scenario, batch size is 1
    pipeline.batch_size = 1
    pipeline.seq_num = 1
    pipeline.token_micro_batch_size = 1
    pipeline.token_micro_batch_num = 1
    pipeline.micro_batch_size = 1

    print("<update_setting> generate_seq_length:", pipeline.generate_seq_length)
    # update args
    args.top_p = pipeline.top_p
    args.temperature = pipeline.temperature
    
    pipeline.change_buffer_size()
    if hasattr(pipeline, 'update_processors'):
        pipeline.update_processors(args)
        

def to_result(
    outputs, tokenizer, top_k_per_token, echo_prompt,
):
    
    i = 0
    n_pads = 0 # in latency inference, #pad should be 0
        
    item = {
        'choices': [], 
    }
    
    for i_ret, output_dict in enumerate(outputs):
        choice = {
            "text": (tokenizer.decode(output_dict['token_ids'][i][n_pads:]) if 'token_ids' in output_dict else ''),
            "index": i_ret,
            "logprobs": {
                "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i][n_pads:] if 'token_ids' in output_dict else [])),
                "token_logprobs": (output_dict['token_logprobs'][i][n_pads:].tolist() if 'token_logprobs' in output_dict else []),
                "top_logprobs": ([
                    {
                        tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                    } \
                    for topk_ids, top_logprobs in zip(
                        output_dict['topk_ids'][i][n_pads:],
                        output_dict['topk_logprobs'][i][n_pads:]
                    )
                ] if top_k_per_token > 0 else None),
                "text_offset": [],
            },
            "finish_reason": "length",
        }
        if echo_prompt:
            if len(choice['logprobs']['token_logprobs']) > 0:
                choice['logprobs']['token_logprobs'][0] = None
                if choice['logprobs']['top_logprobs'] is not None:
                    choice['logprobs']['top_logprobs'][0] = None
        item['choices'].append(choice)
            
    return item


def main():
    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    add_device_arguments(parser)
    add_torch_distributed_inference_w_euler_coordinator_arguments(parser)
    add_inference_arguments(parser)
    add_inference_details_arguments(parser)
    add_global_coordinator_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--net-interface', type=str, default='default', metavar='S',
                        help='network interface name.')
    args = parser.parse_args()
    print_arguments(args)
    # torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    print("Print working directory:", args.working_directory)
    model_name_abbr = args.model_name.split('/')[-1]
    print("model name abbr: ", model_name_abbr)
    print("model name: ", alias_to_model_name(model_name_abbr))
    init_coordinator_client(args, alias_to_model_name(model_name_abbr))
    coord_client = get_coordinator_client()

    pipe = None
    try:
        res = coord_client.notify_inference_join(args.net_interface)
        prime_ip = res['prime_ip']
        rank = res['rank']
        port = res['nccl_port']

        print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")

        init_inference_communicators_with_coordinator(args, prime_ip, rank, port=port)

        if get_pipeline_parallel_rank() == 0:
            coord_client.update_status("running", returned_payload={'state': 'initialized'})

        input_path = coord_client.load_input_job_from_dfs(args.job_id, return_path=True)
        request_processor = get_request_processor(args, infer_data=input_path)
        request_processor.set_arguments(args)

        pipe = get_pp_inference_module(args, device, rank=rank, be_coordinated=False)

        print(f"Inference pipeline loading model <{model_name_abbr}> is done!")
        if get_pipeline_parallel_rank() == 0:
            coord_client.update_status("running", returned_payload={'state': 'model_loaded'})

        if args.profiling == 'no-profiling':
            _ = distributed_inference_mask_iter(args, pipe, device, request_processor, client=coord_client)
        else:
            prefix = './trace_json/inference_' + args.pp_mode
            trace_file = prefix + get_inference_arguments_str(args, rank=rank) + '_' + args.profiling + '_' + \
                         args.trace_postfix + '.json'
            if args.profiling == 'tidy_profiling':
                _ = distributed_inference_mask_iter(args, pipe, device, request_processor, client=coord_client)
                pipe.export_profiling_result(filename=trace_file)
            else:
                print("No recognized profiler?")
                assert False
        if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size()-1:
            coord_client.update_status("finished", returned_payload={'result': request_processor.data})
    except Exception as e:
        print('Exception in batch inference:', e)
        coord_client.update_status("failed", returned_payload={'message': str(e)})

    try:
        local_cord_client = LocalCoordinatorClient(
            working_directory="/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/new/working_dir/",
            coordinator_url="https://coordinator.shift.ml/eth",
        )
        tokenizer = get_tokenizer(args)
        
        begin_time = time.time()
        max_time = 3600
        
        while True:
            
            now = time.time()
            if now - begin_time > max_time:
                logger.info("Reaching max time. Exit interactive mode.")
                break
            else:
                logger.info(f"{now - begin_time} seconds remaining for interactive mode.")
            
            # TODO: please check here
            instructions = local_cord_client.fetch_instructions(alias_to_model_name(model_name_abbr), rank)
            last_instruction = instructions[-1]
            
            if last_instruction["message"] == "break":
                logger.info("Received stop instruction.")
                break
            elif last_instruction["message"] == "continue":
                logger.info("Received keep instruction.")
                sleep(10)
            elif last_instruction["message"] == "run":
                for instruction in [x for x in instructions if x["message"] == "run"]:

                    job_id = None
                    try: 
                        
                        logger.info("Instruction:")
                        logger.info(str(instruction))
                        
                        # TODO: we assume len(payload) is 1, right?
                        query = instruction['payload']['payload'][0]
                        prompt = query['prompt']
                        job_id = instruction['payload']['id']
                        job_status = instruction['payload']['status']
                        
                        if job_status != "submitted":
                            continue

                        # set input length
                        seq_length = tokenizer(
                            prompt, return_tensors='pt', padding=True, truncation=False
                        )['input_ids'].size(1)
                        seq_length = min(seq_length, 2048 - query.get('max_tokens', 1)) # 2048 is hardcoded.
                        
                        logger.info(f"Set input length to {seq_length}.")
                        tokenizer.model_max_length = seq_length
                        pipe.input_seq_length = seq_length

                        # update hyperparameters and buffers
                        logger.info(f"Update settings.")
                        update_setting(args, pipe, query)

                        # get inputs
                        inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, )
                        input_ids = inputs['input_ids'].long().to(device)
                        attention_mask = inputs['attention_mask'].long().to(device)
                    
                        # run inference
                        logger.info(f"Start Inference.")
                        output_ids_list = []
                        pipe.inference_batch(input_ids, output_ids_list, attention_mask=attention_mask)

                        if get_pipeline_parallel_rank() == pipe.pipeline_group_size - 1:
                            result = to_result(output_ids_list, tokenizer, pipe.top_k_per_token, pipe.echo_prompt)
                            return_payload = {
                                'request': query,
                                'result': result,
                            }

                            # TODO: please check if return_payload is correct.
                            local_cord_client.update_status(
                                job_id,
                                "finished",
                                returned_payload=return_payload
                            )
                            
                    except Exception as e:
                        error = traceback.format_exc()
                        local_cord_client.update_status(
                            job_id,
                            "failed",
                            returned_payload={"message": error}
                        )
                        print(error)
            sleep(10)
            
    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == '__main__':
    main()
