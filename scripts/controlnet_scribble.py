from PIL import Image, ImageOps
from pathlib import Path
from tqdm import trange
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import HEDdetector
from generate_descriptions import load_descriptions


if __name__ == '__main__':
    hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    Path('baseline_predictions/').mkdir(exist_ok=True)
    for ind in trange(1250):
        sketch_dir = Path('sampled', f'{ind}')
        sketch_lst = list(sketch_dir.glob('**/*_sketch.png'))
        if len(sketch_lst) == 0:
            continue
        sketch_path = str(sketch_lst[0])
        sketch = Image.open(sketch_path)
        sketch = ImageOps.invert(sketch)
        description = load_descriptions(f'descriptions/{ind}.txt')[0]
        image = pipe(description, sketch, num_inference_steps=20).images[0]
        image.save(f'baseline_predictions/{ind}.png')
