import os
import torch
import random
import argparse
# from loguru import logger
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from coordinator.coordinator_client import LocalCoordinatorClient
from utils.dist_args_utils import *


def main():
    parser = argparse.ArgumentParser(
        description='Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='test',
                        metavar='S', help='Job ID')
    parser.add_argument('--cord_url', type=str, default='localhost',)
    add_global_coordinator_arguments(parser)
    add_lsf_coordinator_arguments(parser)
    args = parser.parse_args()
    print_arguments(args)
    local_cord_client = LocalCoordinatorClient(
        working_directory="./",
        coordinator_url=args.cord_url,
    )
    local_cord_client.update_status(args.job_id, "running")

    output_dir = os.path.join(
        "./",
    )
    print("Loading Stable Diffusion model...")
    lms = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-4",
        scheduler=lms,
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda:0")

    print("Stable Diffusion model loaded.")

    return_msg = local_cord_client.load_input_job_from_dfs(args.job_id)
    if return_msg is not None:
        print(f"Received a new job. {return_msg}")

        job_requests = return_msg

        for job_request in job_requests:
            if isinstance(job_request['input'], str):
                text = [job_request['input']]
                num_return_sequences = [job_request['num_returns']]

            elif isinstance(job_request['input'], list):
                text = job_request['input']
                if isinstance(job_request['num_returns'], int):
                    num_return_sequences = [
                        job_request['num_returns']]*len(text)
                else:
                    num_return_sequences = job_request['num_returns']

            if len(text) != len(num_return_sequences):
                raise ValueError(
                    "The length of text and num_return_sequences (if given as a list) should be the same.")

            results = {}
            results['output'] = []
            with torch.no_grad():
                with autocast("cuda"):
                    img_results = []
                    generated_image_ids = []
                    for i in range(len(text)):
                        for j in range(num_return_sequences[i]):
                            image = pipe(text[i])["sample"][0]
                            # randomly generate a image id
                            image_id = random.randint(0, 1000000)
                            image.save(os.path.join(
                                output_dir, f"{image_id}.png"))
                            generated_image_ids.append(
                                os.path.join(output_dir, f"{image_id}.png"))
                            succ, img_id = local_cord_client.upload_file(
                                os.path.join(output_dir, f"{image_id}.png"))
                            if succ:
                                img_results.append(
                                    "https://planetd.shift.ml/files/"+img_id)
                            else:
                                print("Upload image failed")
                        results["output"].append(img_results)
                    local_cord_client.update_status(
                        args.job_id,
                        "finished",
                        returned_payload=results
                    )
                    # clear cache
                    for image_id in generated_image_ids:
                        os.remove(image_id)

    # now finished the primary job, waiting and keep fetching instructions for next steps
    while True:
        instructions = local_cord_client.fetch_instructions("stable_diffusion")
        print(instructions)
        
if __name__ == '__main__':
    main()
