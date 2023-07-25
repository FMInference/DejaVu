from stable_diffusion_videos import StableDiffusionWalkPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import torch

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    torch_dtype=torch.float16,
    revision="fp16",
    scheduler=LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
).to("cuda")

video_path = pipeline.walk(
    prompts=['a cat', 'a dog'],
    seeds=[42, 1337],
    num_interpolation_steps=3,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='dreams',        # Where images/videos will be saved
    name='animals_test',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
)