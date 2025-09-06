# Script for building a synthetic image dataset via ComfyUI.
# --------------------------------------------------------------------------------

import json
import random
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

import const
from comfyui_adapter import ComfyUIAdapter
import argparse
from datetime import datetime

timestamp_format = "%Y%m%d_%H%M%S"

class PromptBank(BaseModel):
    prefix: str
    suffix: str
    prompts: List[str]

class ImageRecord(BaseModel):
    prompt: str
    prompt_base: Optional[str] = ""
    image_path: str
    seed: int

class ImageSet(BaseModel):
    images: List[ImageRecord] = []
    metadata: dict = {}

def load_prompt_bank(file_path: str) -> PromptBank:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return PromptBank(**data)

def make_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().strftime(timestamp_format)
    out_dir = base_dir / f"dataset_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def write_dataset(dataset: ImageSet, file_path: Path):
    with open(file_path, 'w') as f:
        json.dump(dataset.dict(), f, indent=2)

def build_images(comfy: ComfyUIAdapter,
                 prompt_bank: PromptBank,
                 count: int,
                 out_dir: Path) -> ImageSet:
    dataset = ImageSet()
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    for i in range(count):
        # Pick a random prompt
        prompt = random.choice(prompt_bank.prompts)
        full_prompt = f"{prompt_bank.prefix}{prompt}{prompt_bank.suffix}".strip()
        print(f"Generating image {i + 1}/{count}")
        print(f"Prompt: {full_prompt}")

        # Pick a random seed
        seed_val = random.randint(1, 999999)
        print(f"Using seed: {seed_val}")

        # Generate
        img = comfy.txt2img(prompt=full_prompt, seed=seed_val)

        # Save
        img_path = img_dir / f"image_{i:04d}.png"
        img.save(img_path)

        dataset.images.append(ImageRecord(
            prompt=full_prompt,
            prompt_base=prompt,
            image_path=str(img_path.relative_to(out_dir)),
            seed=seed_val
        ))

        print(f"Saved: {img_path}")
        time.sleep(0.1)

    return dataset

def main(args):
    out_root = Path(args.out)
    prompt_bank = load_prompt_bank(args.prompts)

    comfy = ComfyUIAdapter()
    out_dir = make_output_dir(out_root)
    print(f"Creating dataset at: {out_dir}")

    dataset = build_images(comfy, prompt_bank, args.num_images, out_dir)

    dataset.metadata = {
        "created_at": datetime.now().isoformat(),
        "num_images": len(dataset.images),
        "prompt_bank": args.prompts,
    }

    out_json = out_dir / "dataset.json"
    write_dataset(dataset, out_json)
    print(f"Dataset written to: {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a structured sparse image dataset for text-to-image research")
    parser.add_argument("--prompts", type=str, default="prompts.json", help="Path to the prompt bank JSON file")
    parser.add_argument("--out", type=str, default=const.dir_image_datasets, help="Base directory for generated datasets")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")

    args = parser.parse_args()
    main(args)
