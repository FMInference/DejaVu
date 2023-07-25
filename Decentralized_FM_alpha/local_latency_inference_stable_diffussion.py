import os
from time import sleep
import torch
import random
import argparse
from loguru import logger
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
        working_directory="/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/new/working_dir/",
        coordinator_url=args.cord_url,
    )
    local_cord_client.update_status(args.job_id, "running")

    output_dir = os.path.join(
        "/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/new/working_dir/",
    )
    logger.info("Loading Stable Diffusion model...")
    lms = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=lms,
        use_auth_token=True,
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda:0")

    logger.info("Stable Diffusion model loaded.")

    return_msg = local_cord_client.load_input_job_from_dfs(args.job_id)
    if return_msg is not None:
        logger.info(f"Received a new job. {return_msg}")

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
                                logger.error("Upload image failed")
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

        instructions = local_cord_client.fetch_instructions("stable_diffusion", 0)
        last_instruction = instructions[-1]
        if last_instruction["message"] == "break":
            logger.info("Received stop instruction.")
            break
        elif last_instruction["message"] == "continue":
            logger.info("Received keep instruction.")
            sleep(10)
        elif last_instruction["message"] == "run":
            for instruction in [x for x in instructions if x["message"] == "run"]:
                prompts = instruction['payload']['payload'][0]['input']
                num_of_returns = instruction['payload']['payload'][0]['num_returns']
                job_id = instruction['payload']['id']
                job_status = instruction['payload']['status']
                if job_status == "submitted":
                    if isinstance(prompts, str):
                        prompts = [prompts]
                        num_of_returns = [num_of_returns]

                    elif isinstance(prompts, list):
                        if isinstance(num_of_returns, int):
                            num_of_returns = [num_of_returns]*len(prompts)
                        else:
                            num_of_returns = num_of_returns

                    if len(prompts) != len(num_of_returns):
                        raise ValueError(
                            "The length of text and num_return_sequences (if given as a list) should be the same.")

                    logger.info("received prompt: {}".format(prompts))
                    try:
                        with torch.no_grad():
                            with autocast("cuda"):
                                img_results = []
                                results={"output": []}
                                generated_image_ids = []
                                for i in range(len(prompts)):
                                    for j in range(num_of_returns[i]):
                                        image = pipe(prompts[i])["sample"][0]
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
                                            logger.error("Upload image failed")
                                    results["output"].append(img_results)
                                local_cord_client.update_status(
                                    job_id,
                                    "finished",
                                    returned_payload=results
                                )
                                # clear cache
                                for image_id in generated_image_ids:
                                    os.remove(image_id)
                    except Exception as e:
                        logger.error(e)
                        local_cord_client.update_status(
                            job_id,
                            "failed",
                            returned_payload={"message": str(e)}
                        )
        sleep(10)
if __name__ == '__main__':
    main()
