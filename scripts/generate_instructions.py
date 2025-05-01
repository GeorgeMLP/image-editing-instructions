import os
from pathlib import Path
from tqdm import trange
import torch  # version <= 2.6, torch 2.7 doesn't work
from torch import LongTensor
from PIL import Image, ImageFile
from transformers import TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor


prompt = (
    "We are building a dataset for image editing. Given a pair of images "
    "(a source image and a target image), please generate 5 different natural "
    "language instructions that describe how to edit the first image to "
    "transform it into the second image. Example instructions: \"add a dog on "
    "the bench\", \"draw an airplane in the sky\", etc.\n\n"
    "Guidelines:\n"
    "- Each instruction should describe a single, clear editing operation.\n"
    "- Focus on visual, content-related changes (e.g., adding, removing, or "
    "modifying objects, changing colors, adjusting backgrounds, etc.).\n"
    "- Write one instruction per line.\n"
    "- Do not output anything other than the 5 instructions.\n\n"
    "The source and target images are given as follows. Make sure your "
    "instructions accurately describe their differences.\n"
)


def save_instructions(instructions: list[str], save_path: str | Path) -> None:
    with open(save_path, 'w') as f:
        f.writelines([s + '\n' for s in instructions])


def load_instructions(save_path: str | Path) -> list[str]:
    with open(save_path, 'r') as f:
        instructions = f.readlines()
    return [s.strip() for s in instructions]


def generate_and_save_instructions(
    image_before: ImageFile,
    image_after: ImageFile,
    save_path: str | Path | None = None,
) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        },
        {
            "role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_before},
                {"type": "image", "image": image_after},
            ]
        },
    ]
    inputs: BatchFeature = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to("cuda")

    output: LongTensor = model.generate(**inputs, max_new_tokens=500)
    response: str = processor.decode(output[0], skip_special_tokens=True)
    instructions: list[str] = list(filter(None, response.split('\n')))[-5:]
    if save_path is not None:
        save_instructions(instructions, save_path)
    return instructions


if __name__ == '__main__':
    quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor: Gemma3Processor = AutoProcessor.from_pretrained(
        "google/gemma-3-27b-it",
        padding_side="left",
        use_fast=True,
    )

    # demo
    # image_before = Image.open('example-images/1.png')
    # image_after = Image.open('example-images/2.jpg')
    # save_path = 'example-images/instructions.txt'
    # generate_and_save_instructions(image_before, image_after, save_path)
    # print(load_instructions(save_path))

    # generate instructions for the whole dataset
    Path('instructions/').mkdir(exist_ok=True)
    for ind in trange(1250):
        save_path = f'instructions/{ind}.txt'
        if os.path.exists(save_path):  # already generated
            continue
        photo_dir = Path('sampled', f'{ind}')
        removed_dir = Path('inpainted', f'{ind}')
        photo_lst = list(photo_dir.glob('**/*_photo.jpg'))
        removed_lst = list(removed_dir.glob('**/*_remove.jpg'))
        if len(photo_lst) == 0 or len(removed_lst) == 0:
            continue
        photo_path = str(photo_lst[0])
        removed_path = str(removed_lst[0])
        image_before = Image.open(removed_path)
        image_after = Image.open(photo_path)
        generate_and_save_instructions(image_before, image_after, save_path)
