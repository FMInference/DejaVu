import os
import torch
import random
import argparse
#from loguru import logger
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

if __name__ == '__main__':
  
    print('Begin of StableDiffusion')

    lms = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )
    
    print('Middle of StableDiffusion')

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-4",
        scheduler=lms,
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda:0")

    with torch.no_grad():
        with autocast("cuda"):
            image = pipe('a dragon in the sky')["sample"][0]

    image.save('./out.png')
    
    print('End of StableDiffusion')
