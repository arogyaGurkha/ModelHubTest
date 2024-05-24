from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from skbio.diversity.alpha import simpson
import numpy as np
from scipy.special import kl_div

DEVICE = "cuda:0"


def init_feature_extractor():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.to(DEVICE)
    model.eval()
    return processor, model

def calculate_kl(dataset_labels, sampled_dataset_labels):
    original_counts = np.bincount(dataset_labels)
    sampled_counts = np.bincount(sampled_dataset_labels)

    original_prob = original_counts / original_counts.sum()
    sampled_prob = sampled_counts / sampled_counts.sum()

    max_length = max(len(original_prob), len(sampled_prob))
    original_prob = np.pad(
        original_prob, (0, max_length - len(original_prob)), "constant"
    )
    sampled_prob = np.pad(sampled_prob, (0, max_length - len(sampled_prob)), "constant")

    kl_divergence = np.sum(kl_div(original_prob, sampled_prob))

    return kl_divergence



def convert_to_hf_dataset(dataset, sampled_indices):
    sampled_data = [dataset[idx] for idx in sampled_indices]
    hf_dataset = Dataset.from_dict(
        {key: [d[key] for d in sampled_data] for key in sampled_data[0]}
    )
    return hf_dataset


def predict_batch(processor, model, batch):
    images = [
        image.convert("RGB") if image.mode != "RGB" else image
        for image in batch.values()
    ]
    inputs = processor(images=images, return_tensors="pt")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)

    embeddings_dict = {
        idx: embedding for idx, embedding in zip(batch.keys(), cls_embeddings)
    }
    return embeddings_dict


def batched_dataset(dataset, batch_size):
    batch = {}
    for idx, item in enumerate(dataset):
        batch[idx] = item["image"]
        if len(batch) == batch_size:
            yield batch
            batch = {}

    if batch:
        yield batch


def create_dataset(n_samples=3618):
    dataset = load_dataset(
        "imagefolder",
        data_dir="/workspaces/ModelHubTest/src/data/assets/cat_vs_dog",
        split="test",
    )
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset


def add_embedding_if_representative(
    existing_embeddings, existing_indices, new_embedding, img_idx, threshold=0.6
):
    new_embedding = new_embedding.to(DEVICE)
    if existing_embeddings.nelement() == 0:
        existing_embeddings = new_embedding.unsqueeze(0)
        existing_indices.append(img_idx)
        return True, existing_embeddings, existing_indices

    similarities = torch.matmul(existing_embeddings, new_embedding.unsqueeze(1))

    if torch.all(similarities < threshold):
        existing_embeddings = torch.cat(
            [existing_embeddings, new_embedding.unsqueeze(0)], dim=0
        )
        existing_indices.append(img_idx)
        return True, existing_embeddings, existing_indices
    return False, existing_embeddings, existing_indices


def display_images(dataset, img_indices):
    print(f"Displaying {len(img_indices)} images!")
    num_images = len(img_indices)
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()

    for idx, image_idx in enumerate(img_indices):
        image = dataset[image_idx]["image"]
        axes[idx].imshow(image)
        axes[idx].set_title(f"Image Index: {image_idx}")
        axes[idx].axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def create_embeddings_for_sampling():
    embeddings_dict = {}
    batch_size = 128
    processor, model = init_feature_extractor()
    dataset = create_dataset()
    for batch in batched_dataset(dataset, batch_size):
        result = predict_batch(processor, model, batch)
        embeddings_dict.update(result)
    return dataset, embeddings_dict


if __name__ == "__main__":
    s_t = 0.06
    embeddings_dict = {}
    batch_size = 128

    processor, model = init_feature_extractor()
    dataset = create_dataset()
    for batch in batched_dataset(dataset, batch_size):
        result = predict_batch(processor, model, batch)
        embeddings_dict.update(result)

    repr_embeddings = torch.empty((0, 768), device=DEVICE)
    repr_indices = []

    for idx, new_embedding in embeddings_dict.items():
        was_added, repr_embeddings, repr_indices = add_embedding_if_representative(
            repr_embeddings, repr_indices, new_embedding, idx, threshold=s_t
        )

    sampled_dataset = convert_to_hf_dataset(dataset, repr_indices)
    print(f"Simpson's Index: {simpson(np.bincount(sampled_dataset['label']))}")
    print(f"KL Divergence: {calculate_kl(dataset['label'], sampled_dataset['label'])}")

    display_images(dataset, repr_indices)
