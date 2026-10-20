from tutus.utils import conversation
from time import time
import pandas as pd
from PIL import Image
import torch
import os
from rich.progress import track
from tutus.utils.dist import get_device_id


@torch.no_grad()
def generate_predictions(model, processor, prompts, images, use_distributed=False):
    if isinstance(images[0], str):
        loaded_images = [[Image.open(image_path).convert("RGB")] for image_path in images]
    elif isinstance(images[0], Image.Image):
        loaded_images = images
    elif isinstance(images[0], list) and isinstance(images[0][0], Image.Image):
        loaded_images = images
    else:
        raise ValueError(f"Images should be a list of file paths or PIL Image objects, but got {type(images[0])}")

    loaded_prompts = [processor.apply_chat_template(conversation(**prompt), add_generation_prompt=True) for prompt in prompts]
    inputs = processor(images=loaded_images, text=loaded_prompts, return_tensors="pt", padding=True).to(get_device_id(), dtype=torch.bfloat16)
    
    if use_distributed and hasattr(model, "module"):
        outputs = model.module.generate(**inputs, max_new_tokens=64)
    else:
        outputs = model.generate(**inputs, max_new_tokens=64)
        
    outputs_processed = processor.batch_decode(outputs, skip_special_tokens=True)

    return [r"{}".format(output) for output in outputs_processed]


