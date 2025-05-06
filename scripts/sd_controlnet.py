from PIL import Image, ImageOps
from pathlib import Path
from tqdm import trange
import torch
import cv2
import numpy as np
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from generate_instructions import load_instructions


def sketch_to_filled_mask(
    sketch_path: str,
    mask_path: str,
    thresh_value: int = 200,
    closing_kernel_size: int = 15,
) -> None:
    # Load and grayscale
    img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {sketch_path}")

    # Invert + threshold: lines become white on black
    _, th = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to bridge gaps
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw filled contours into mask
    mask = np.zeros_like(closed)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Save result
    cv2.imwrite(mask_path, mask)


if __name__ == '__main__':
    scribble_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=scribble_controlnet,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    )
    pipe.vae = vae

    Path('masks/').mkdir(exist_ok=True)
    Path('sd_baseline_predictions/').mkdir(exist_ok=True)
    for ind in trange(1250):
        image_dir = Path('inpainted', f'{ind}')
        scribble_dir = Path('sampled', f'{ind}')
        image_lst = list(image_dir.glob('**/*_remove.jpg'))
        scribble_lst = list(scribble_dir.glob('**/*_sketch.png'))
        if len(image_lst) == 0 or len(scribble_lst) == 0:
            continue
        image_path = str(image_lst[0])
        scribble_path = str(scribble_lst[0])

        sketch_to_filled_mask(
            sketch_path=scribble_path,
            mask_path=f'masks/{ind}.png',
        )

        image = Image.open(image_path).convert('RGB')
        scribble = ImageOps.invert(Image.open(scribble_path)).convert('RGB')
        mask = Image.open(f'masks/{ind}.png').convert("L")
        prompt = load_instructions(f'instructions/{ind}.txt')[0]

        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            control_image=scribble,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
        )
        result.images[0].save(f'sd_baseline_predictions/{ind}.png')
