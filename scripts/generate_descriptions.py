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
    "Please generate a detailed description of the following image. Generate "
    "the description on one line and do not output anything other than the "
    "description.\n"
)


def save_descriptions(descriptions: list[str], save_path: str | Path) -> None:
    with open(save_path, 'w') as f:
        f.writelines([s + '\n' for s in descriptions])


def load_descriptions(save_path: str | Path) -> list[str]:
    with open(save_path, 'r') as f:
        descriptions = f.readlines()
    return [s.strip() for s in descriptions]


def generate_and_save_descriptions(
    image: ImageFile,
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
                {"type": "image", "image": image},
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
    instructions: list[str] = list(filter(None, response.split('\n')))[-1:]
    if save_path is not None:
        save_descriptions(instructions, save_path)
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
    # image = Image.open('example-images/2.jpg')
    # save_path = 'example-images/descriptions.txt'
    # generate_and_save_descriptions(image, save_path)
    # print(load_descriptions(save_path))

    # generate instructions for the whole dataset
    Path('descriptions/').mkdir(exist_ok=True)
    for ind in trange(1250):
        save_path = f'descriptions/{ind}.txt'
        if os.path.exists(save_path):  # already generated
            continue
        photo_dir = Path('sampled', f'{ind}')
        photo_lst = list(photo_dir.glob('**/*_photo.jpg'))
        if len(photo_lst) == 0:
            continue
        photo_path = str(photo_lst[0])
        image = Image.open(photo_path)
        generate_and_save_descriptions(image, save_path)
