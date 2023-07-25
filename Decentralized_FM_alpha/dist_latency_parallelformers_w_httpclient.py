import torch
from transformers import OPTForCausalLM, AutoTokenizer, OPTConfig
from parallelformers import parallelize
import argparse
from coordinator.http_coordinate_client import alias_to_model_name
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
from time import sleep


def to_result(output_dict, tokenizer):
    # TODO, Lots of missing attributes here!!!!
    item = {'choices': [], }
    choice = {
        "text": (tokenizer.decode(output_dict['sequences'][0])),
        "index": 0,
        "finish_reason": "length",
    }
    item['choices'].append(choice)
    return item


def main():
    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--model-name', type=str, default='./pretrained_models/gpt2', metavar='S',
                        help='trained model path')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')
    args = parser.parse_args()

    print("Print working directory:", args.working_directory)
    model_name_abbr = args.model_name.split('/')[-1]
    print("model name abbr: ", model_name_abbr)
    print("model name: ", alias_to_model_name(model_name_abbr))

    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        config = OPTConfig.from_pretrained(args.model_name)
        model = OPTForCausalLM(config)
        parallelize(model, num_gpus=args.world_size, fp16=True, verbose='detail')
        torch.cuda.empty_cache()
        local_cord_client.update_status(args.job_id, "running")
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    try:
        while True:
            instructions = local_cord_client.fetch_instructions(alias_to_model_name(model_name_abbr), 0)
            last_instruction = instructions[-1]

            if last_instruction["message"] == "break":
                logger.info("Received stop instruction.")
                logger.info("# BREAK ")
                break
            elif last_instruction["message"] == "continue":
                logger.info("Received keep instruction.")
                sleep(10)
            elif last_instruction["message"] == "run":

                fetched_tasks = [x for x in instructions
                                 if x["message"] == "run" and x['payload']['status'] == 'submitted']

                for instruction in fetched_tasks:
                    job_id = None
                    try:
                        logger.info("Instruction:")
                        logger.info(str(instruction))
                        # TODO: we assume len(payload) is 1, right?
                        query = instruction['payload']['payload'][0]
                        prompt = query['prompt']
                        job_id = instruction['payload']['id']
                        print(f"Job <{job_id}> starts to run.")

                        with torch.no_grad():
                            current_input = tokenizer(prompt,  padding='max_length', return_tensors='pt')
                            input_ids = current_input['input_ids'].long().cuda()
                            output_ids = model.generate(input_ids,
                                                        max_new_tokens=query.get('max_tokens', 1),
                                                        do_sample=True,
                                                        temperature=query.get('temperature', 0),
                                                        top_p=query.get('top_p', 1.0),
                                                        num_return_sequences=query.get('n', 1))
                            result = to_result(output_ids, tokenizer)
                            return_payload = {
                                'request': query,
                                'result': result,
                            }
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
                    sleep(1)

    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == '__main__':
    main()
