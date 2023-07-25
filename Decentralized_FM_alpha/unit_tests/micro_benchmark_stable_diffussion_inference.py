import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler



def main():
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

    text = ['a lovely dog with sunglasses']

    with torch.no_grad():
        with autocast("cuda"):
            for i in range(5):
                image = pipe(text)["sample"][0]
                image.save(f'./dog-{i}.jpeg', quality=95)


if __name__ == '__main__':
    main()