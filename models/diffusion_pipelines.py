import torch
from diffusers import StableDiffusionPipeline


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
checkpoints = [
    "runwayml/stable-diffusion-v1-5"
]
pipeline = StableDiffusionPipeline.from_pretrained(checkpoints[0])
pipeline = pipeline.to(device)

print("Completed loading the model and setting the device.")