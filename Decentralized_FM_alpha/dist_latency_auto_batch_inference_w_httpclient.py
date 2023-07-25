from time import sleep
import argparse
from loguru import logger
from pipeline_parallel.dist_pipeline_inference_batch_auto_mask_sample_token_pipe \
    import DistInferenceMaskTokenPipeAutoBatch
from utils.dist_args_utils import *
from comm.comm_utils import *
from coordinator.http_coordinate_client import alias_to_model_name
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
import math


def to_result(output_dict, tokenizer, top_k_per_token, echo_prompt):
    n_pads = 0  # in latency inference, #pad should be 0
    item = {'choices': [], }
    choice = {
        "text": (tokenizer.decode(output_dict['token_ids'][0][n_pads:]) if 'token_ids' in output_dict else ''),
        "index": 0,
        "logprobs": {
            "tokens": (tokenizer.convert_ids_to_tokens(
                output_dict['token_ids'][0][n_pads:] if 'token_ids' in output_dict else [])),
            "token_logprobs": (
                output_dict['token_logprobs'][0][n_pads:].tolist() if 'token_logprobs' in output_dict else []),
            "top_logprobs": ([{tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item() for
                              topk_id, top_logprob in zip(topk_ids, top_logprobs)}
                             for topk_ids, top_logprobs in zip(output_dict['topk_ids'][0][n_pads:],
                                                               output_dict['topk_logprobs'][0][n_pads:])
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
    parser.add_argument('--auto-batch-size', type=int, default=4, metavar='S',
                        help='auto batched size (default: 4)')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
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

    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )

    pipe = None
    rank = None
    try:
        res = local_cord_client.notify_inference_join(args.job_id, args.net_interface)
        prime_ip = res['prime_ip']
        rank = res['rank']
        port = res['nccl_port']
        print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")
        init_inference_communicators_with_coordinator(args, prime_ip, rank, port=port)

        if get_pipeline_parallel_rank() == 0:
            local_cord_client.update_status(args.job_id, "running", returned_payload={'state': 'initialized'})

        pipe = DistInferenceMaskTokenPipeAutoBatch(args, device)

        print(f"Inference pipeline loading model <{model_name_abbr}> is done!")
        if get_pipeline_parallel_rank() == 0:
            local_cord_client.update_status(args.job_id, "running", returned_payload={'state': 'model_loaded'})

    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    try:
        tokenizer = pipe.tokenizer
        while True:
            try:
                task_settings = None
                input_ids = None
                job_ids = None
                output_ids_list = None
                attention_masks = None
                if get_pipeline_parallel_rank() == 0:
                    instructions = local_cord_client.fetch_instructions(alias_to_model_name(model_name_abbr), rank)
                    last_instruction = instructions[-1]

                    if last_instruction["message"] == "break":
                        logger.info("Received stop instruction.")
                        logger.info("# BREAK ")
                        break
                    elif last_instruction["message"] == "continue":
                        logger.info("Received keep instruction.")
                        pipe.has_work[:] = 0
                    elif last_instruction["message"] == "run":
                        fetched_tasks = [x for x in instructions
                                         if x["message"] == "run" and x['payload']['status'] == 'submitted']

                        batch_size = max(len(fetched_tasks), args.auto_batch_size)
                        current_tasks = fetched_tasks[:batch_size]

                        task_settings = []
                        input_ids = []
                        job_ids = []

                        for instruction in current_tasks:
                            logger.info("Instruction:")
                            logger.info(str(instruction))
                            # TODO: we assume len(payload) is 1, right?
                            query = instruction['payload']['payload'][0]
                            prompt = query['prompt']
                            job_id = instruction['payload']['id']
                            print(f"Job <{job_id}> has been batched")
                            job_ids.append(job_id)

                            task_settings.append(query)
                            current_input = tokenizer(prompt, return_tensors='pt', padding='max_length',
                                                      truncation=True)
                            current_input_ids = current_input['input_ids'].long().to(device)
                            input_ids.append(current_input_ids)
                        if len(job_ids) > 0:
                            pipe.has_work[:] = 1
                elif get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
                    output_ids_list = []

                # torch.distributed.barrier() This is a bad practice, which leads to timeout.
                pipe.sync_has_work()

                if pipe.has_work[:] == 1:
                    pipe.update_batch_setting(task_settings=task_settings, job_ids=job_ids)
                    pipe.inference_batch(input_ids, output_ids_list, attention_mask=attention_masks)

                    if get_pipeline_parallel_rank() == pipe.pipeline_group_size - 1:
                        job_ids = pipe.current_job_ids
                        for i in range(len(job_ids)):
                            print(output_ids_list[i])
                            result = to_result(output_ids_list[i], tokenizer, pipe.top_k_per_token[i],
                                               pipe.echo_prompt[i])
                            return_payload = {
                                'request': pipe.task_settings[i],
                                'result': result,
                            }

                            local_cord_client.update_status(
                                job_ids[i],
                                "finished",
                                returned_payload=return_payload
                            )
                    pipe.has_work[:] = 0
                else:
                    sleep(10)

            except Exception as e:
                error = traceback.format_exc()
                for job_id in pipe.current_job_ids:
                    local_cord_client.update_status(
                        job_id,
                        "failed",
                        returned_payload={"message": error}
                    )
                print(error)
            sleep(1)

    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == '__main__':
    main()
