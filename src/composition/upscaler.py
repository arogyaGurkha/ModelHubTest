"""
Script to pre-upscale dataset images.
"""

import torch
from diffusers import StableDiffusionUpscalePipeline
from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm


def create_dataset(filepath, n_samples=5849):
    dataset = load_dataset(
        "imagefolder",
        data_dir=filepath,
        split="test",
    )
    # Optionally shuffle and select a subset of the dataset
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset


def get_inputs(prompt, batch_size=1):
    prompts = batch_size * [prompt]
    return {"prompt": prompts}


def save_upscaled_images(upscaled_images, labels, output_dir, batch_id):
    """
    Save the upscaled images to the specified output directory.

    Args:
        upscaled_images (list): List of upscaled PIL images.
        labels (list): List of labels corresponding to each image.
        output_dir (str): Directory where images will be saved.
        batch_id (int): ID of the current batch to ensure unique filenames.
    """
    print(f"Saving {len(upscaled_images)} images...")
    for idx, (img, label) in enumerate(zip(upscaled_images, labels)):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(
            label_dir, f"upscaled_image_batch{batch_id}_img{idx}.png"
        )
        print(f"Saving to {img_path}...")
        img.save(img_path)


def upscaling(dataset, storage_path, gpu_id, batch_size=8, prompt="a vehicle"):
    """
    Upscale images in the dataset using a Stable Diffusion pipeline.

    Args:
        dataset (dict): Dictionary containing 'image' and 'label' keys with corresponding lists.
        storage_path (str): Path where upscaled images will be stored.
        gpu_id (int): ID of the GPU to use for upscaling.
        batch_size (int, optional): Number of images to process in each batch. Defaults to 8.
        prompt (str, optional): Prompt to use for the upscaling process. Defaults to "a vehicle".

    Returns:
        list: List of upscaled images.
    """
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(gpu_id)

    upscaled_images = []

    for batch_id, i in enumerate(tqdm(range(0, len(dataset["image"]), batch_size))):
        batch_images = dataset["image"][i : i + batch_size]
        batch_labels = dataset["label"][i : i + batch_size]

        inputs = get_inputs(prompt=prompt, batch_size=len(batch_images))
        batch_upscaled_images = pipeline(**inputs, image=batch_images).images

        save_upscaled_images(
            batch_upscaled_images, batch_labels, storage_path, batch_id
        )

        upscaled_images.extend(batch_upscaled_images)

    return upscaled_images


if __name__ == "__main__":
    vehicle_dataset = create_dataset("/workspaces/ModelHubTest/src/dataset/vehicles")
    upscaling(
        vehicle_dataset,
        "/workspaces/ModelHubTest/src/dataset/upscaled_vehicle_dataset/test/",
        gpu_id=1,
        batch_size=16,
    )
