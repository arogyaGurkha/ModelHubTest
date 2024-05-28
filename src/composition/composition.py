from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    pipeline,
    ViTImageProcessor,
    ViTModel,
)
import numpy as np
import matplotlib.pyplot as plt
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
from datasets import load_dataset, Dataset
import argparse
import pandas as pd
import timm
from datetime import datetime
from pytz import timezone
import json
import os
from tqdm.auto import tqdm
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import timm.data
import urllib.request
from sklearn.cluster import KMeans
import random
from skbio.diversity.alpha import simpson
from scipy.special import kl_div
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MultiLabelBinarizer,
    LabelEncoder,
)
from transformers.pipelines.pt_utils import KeyDataset
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

IMAGENET_LABELS = None


class CLS_Models:
    def __init__(self, csv_filepath):
        self.models = self.get_models_from_csv(csv_filepath)
        self.inference_models = None
        self.test_models = None

    def __str__(self):
        return (
            f"All Classification Models:\n{self.models['model']},\n"
            f"Inference Models:\n{self.inference_models['model']},\n"
            f"Test Models:\n{self.test_models['model']},\n"
        )

    def get_models_from_csv(self, filepath):
        clf_models_df = pd.read_csv(filepath)
        return clf_models_df

    def random_sample_inference_models(self, count):
        self.inference_models = self.models.sample(count, random_state=42)
        self.test_models = self.models.drop(self.inference_models.index)


class ExperimentConfig:
    def __init__(
        self,
        runs,
        root_dir,
        description,
        gpu_id,
        is_gt,
        clf_csv_path,
        dataset_path,
        n_samples,
        m_samples,
        batch_processing,
        sampling,
    ):
        self.success = False
        self.runs = runs
        self.root_dir = root_dir
        self.output_dir = root_dir
        self.description = description
        self.gpu_id = gpu_id
        self.n_samples = n_samples
        self.m_samples = m_samples
        self.experiment_time = self.get_current_time()
        self.is_gt = is_gt
        self.clf_csv_path = clf_csv_path
        self.dataset_path = dataset_path
        self.batch_processing = batch_processing
        self.sampling = sampling

    def __str__(self):
        return f"Experiment Description:{self.description},\n"

    def save_experiment_config(self):
        config = {
            "success": self.success,
            "experiment_time": self.experiment_time,
            "description": self.description,
            "is_gt": self.is_gt,
            "n_samples": self.n_samples,
            "m_samples": self.m_samples,
            "iterations": self.runs,
            "output_dir": self.output_dir,
        }
        filename = f"{self.output_dir}/config.json"

        try:
            with open(filename, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Experiment configuration saved at {self.output_dir}")
        except IOError as e:
            print(f"Failed to write to {self.output_dir}: {e}")

    def get_current_time(self):
        utc_9 = timezone("Asia/Tokyo")
        utc_time = datetime.now().astimezone(utc_9)
        return f"{utc_time.hour}:{utc_time.minute}:{utc_time.second}"


def calculate_density(embeddings_dict, n_neighbors=5):
    embeddings = np.stack(
        [embedding.cpu().numpy() for embedding in embeddings_dict.values()]
    )
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(
        embeddings
    )
    distances, _ = neighbors.kneighbors(embeddings)
    densities = np.mean(distances, axis=1)
    density_scores = {idx: densities[i] for i, idx in enumerate(embeddings_dict.keys())}
    return density_scores


def predict_batch(processor, model, batch, gpu_id):
    images = [
        image.convert("RGB") if image.mode != "RGB" else image
        for image in batch.values()
    ]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(f"cuda:{gpu_id}") for k, v in inputs.items()}
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


def init_feature_extractor(gpu_id):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.to(f"cuda:{gpu_id}")
    model.eval()
    return processor, model


def convert_to_hf_dataset(dataset, sampled_indices):
    sampled_data = [dataset[idx] for idx in sampled_indices]
    hf_dataset = Dataset.from_dict(
        {key: [d[key] for d in sampled_data] for key in sampled_data[0]}
    )
    return hf_dataset


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


def kmeans_sampling(embeddings, n_clusters=10, n_samples=100):
    embedding_array = np.array([embedding.cpu() for embedding in embeddings.values()])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding_array)
    cluster_indices = {i: [] for i in range(n_clusters)}
    for idx, label in zip(embeddings.keys(), kmeans.labels_):
        cluster_indices[label].append(idx)

    samples_per_cluster = n_samples // n_clusters
    sampled_indices = []
    for indices in cluster_indices.values():
        if len(indices) >= samples_per_cluster:
            sampled_indices.extend(random.sample(indices, samples_per_cluster))
        else:
            sampled_indices.extend(indices)

    sampled_embeddings = {idx: embeddings[idx] for idx in sampled_indices}
    return sampled_embeddings, sampled_indices


def random_sampling(embeddings, n_samples=100):
    sampled_indices = random.sample(list(embeddings.keys()), n_samples)
    sampled_embeddings = {idx: embeddings[idx] for idx in sampled_indices}
    return sampled_embeddings, sampled_indices


def preprocessing(data):
    print("Preprocessing dataset...")
    data = data.copy()
    data["dataset"] = data["dataset"].apply(lambda x: ",".join(sorted(x)))

    data = data.drop("is_timm", axis=1)

    if "inference_accuracy" in data.columns:
        features = data.drop("inference_accuracy", axis=1)
        label = data["inference_accuracy"]
    else:
        # For new data with no labels (groundt_accuracy)
        features = data
        label = None

    return features, label


def configure_pipeline():
    numeric_features = ["accuracy", "likes", "downloads"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_features = ["base_model", "dataset"]
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def cross_validation(pipeline, X, y, k_folds, scoring="neg_mean_squared_error"):
    scores = cross_val_score(pipeline, X, y, cv=k_folds, scoring=scoring)
    return scores


def predict_accuracy(pipeline, new_data):
    # print("New data shape:", new_data.shape)
    # print("New data columns:", new_data.columns)
    try:
        new_preds = pipeline.predict(new_data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return [None]  # or handle the error as appropriate for your use case

    return new_preds


def get_imagenet_labels():
    # url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    filename = "/workspaces/ModelHubTest/src/composition/imagenet_classes.txt"
    # urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        return [s.strip() for s in f.readlines()]


def map_labels_for_accuracy(labels, predictions, output_dir):
    log_file = f"{output_dir}/accuracy_log.txt"
    logging.basicConfig(
        filename=log_file, filemode="w", level=logging.INFO, format="%(message)s"
    )

    # logging.info(f"Composition result: {predictions}.")
    # logging.info(f"Ground truth label: {labels}")

    label_map = {
        "bicycle": 0,
        "mountain bike, all-terrain bike, off-roader": 0,
        "mountain bike": 0,
        "pickup_truck": 1,
        "pickup, pickup truck": 1,
        "pickup": 1,
        "streetcar": 2,
        "streetcar, tram, tramcar, trolley, trolley car": 2,
        "tank": 3,
        "tank, army tank, armored combat vehicle, armoured combat vehicle": 3,
        "tractor": 4,
        "thresher, thrasher, threshing machine": 4,
        "thresher": 4,
    }

    correct_pred_count = 0
    total_predictions = 0

    for idx, preds in enumerate(predictions):
        if not preds:
            continue

        max_score_prediction = max(preds, key=lambda x: x["score"])
        if max_score_prediction["label"] in label_map:
            if label_map[max_score_prediction["label"]] == labels[idx]:
                correct_pred_count += 1
                # logging.info(
                #     f"{max_score_prediction['label']} was the correct label. True label: {labels[idx]}"
                # )
            else:
                # logging.warning(
                #     f"{max_score_prediction['label']} was the incorrect label. True label: {labels[idx]}"
                # )
                continue
        else:
            logging.warning(f"{max_score_prediction['label']} not in label map.")
        total_predictions += 1

    accuracy = correct_pred_count / total_predictions if total_predictions else 0
    return accuracy


def upscaling(image, gpu_id, prompt="a vehicle"):
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipeline = pipeline.to(f"cuda:{gpu_id}")

    upscaled_image = pipeline(prompt=prompt, image=image).images[0]
    return upscaled_image


def segmentation(image, relevant_labels, gpu_id):
    # TODO: Multiple detection models
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-101", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-101", revision="no_timm"
    ).to(f"cuda:{gpu_id}")

    inputs = processor(images=image, return_tensors="pt").to(f"cuda:{gpu_id}")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.6
    )[0]

    detected_patches = []

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        if str(label.item()) in relevant_labels:
            # print(
            #     f"Detected {model.config.id2label[label.item()]} with confidence "
            #     f"{round(score.item(), 3)} at location {box}"
            # )
            # Extract the sub-patch from the image
            left, top, right, bottom = box
            sub_patch = image.crop((left, top, right, bottom))
            detected_patches.append((sub_patch, score.item(), label.item(), box))

    return detected_patches


def classification(image, model_id, gpu_id):
    print(f"Beginning classification for {model_id}.")
    classifier = pipeline(model=model_id, device=gpu_id)
    return classifier(image)


def batch_segmentation(images, relevant_labels, gpu_id, batch_size=4):
    print("Beginning batch segmentation.")
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-101", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-101", revision="no_timm"
    ).to(f"cuda:{gpu_id}")

    all_detected_patches = []
    num_batches = len(images) // batch_size + int(len(images) % batch_size > 0)

    for i in tqdm(range(num_batches), total=num_batches):
        batch_images = images[i * batch_size : (i + 1) * batch_size]
        processed_inputs = processor(images=batch_images, return_tensors="pt").to(
            f"cuda:{gpu_id}"
        )

        with torch.no_grad():
            outputs = model(**processed_inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in batch_images]).to(
            f"cuda:{gpu_id}"
        )
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.6
        )

        for j, result in enumerate(results):
            image = batch_images[j]
            detected_patches = []

            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                box = [round(i, 2) for i in box.tolist()]
                if str(label.item()) in relevant_labels:
                    # print(
                    #     f"Detected {model.config.id2label[label.item()]} with confidence "
                    #     f"{round(score.item(), 3)} at location {box}"
                    # )
                    # Extract the sub-patch from the image
                    left, top, right, bottom = box
                    sub_patch = image.crop((left, top, right, bottom))
                    detected_patches.append(
                        (sub_patch, score.item(), label.item(), box)
                    )

            all_detected_patches.append(detected_patches)

    return all_detected_patches


def batch_classification(images, model_id, gpu_id, batch_size=128):
    print(f"Beginning batch classification for {model_id}.")
    filtered_images = [
        (index, img) for index, img in enumerate(images) if img is not None
    ]
    filtered_indices, valid_images = zip(*filtered_images)

    classifier = pipeline(model=model_id, device=gpu_id)
    predictions = [None] * len(
        images
    )  # Initialize with None to match the original length

    # Perform classification on valid images
    for pred, index in zip(
        tqdm(
            classifier(list(valid_images), batch_size=batch_size),
            total=len(valid_images),
        ),
        filtered_indices,
    ):
        predictions[index] = pred

    return predictions


def timm_classification(image, model_id, gpu_id):
    # print(IMAGENET_LABELS)
    print(f"Beginning classification for {model_id}.")
    model = timm.create_model(model_id, pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    input_tensor = transforms(image).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        model = model.to(device)
        input_tensor = input_tensor.to(device)
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    predictions = []
    for i in range(top5_prob.size(0)):
        label = (
            IMAGENET_LABELS[top5_catid[i]]
            if top5_catid[i] < len(IMAGENET_LABELS)
            else "unknown"
        )
        predictions.append({"label": label, "score": top5_prob[i].item()})

    print(predictions)
    return predictions


def batch_timm_classification(images, model_id, gpu_id, batch_size=128):
    print(f"Beginning batch classification for {model_id}.")
    model = timm.create_model(model_id, pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    filtered_images = [
        (index, img) for index, img in enumerate(images) if img is not None
    ]
    if not filtered_images:
        return []

    filtered_indices, valid_images = zip(*filtered_images)
    transformed_images = [transforms(image).unsqueeze(0) for image in valid_images]
    input_tensor = torch.cat(transformed_images, dim=0)

    predictions = [None] * len(
        images
    )  # Initialize with None to match the original length

    with torch.no_grad():
        for i in tqdm(
            range(0, len(input_tensor), batch_size),
            total=(len(input_tensor) + batch_size - 1) // batch_size,
        ):
            batch_inputs = input_tensor[i : i + batch_size].to(device)
            output = model(batch_inputs)
            probabilities = torch.nn.functional.softmax(output, dim=1)

            top5_prob, top5_catid = torch.topk(probabilities, 5)

            for j in range(batch_inputs.size(0)):
                preds = []
                for k in range(top5_prob.size(1)):
                    label = (
                        IMAGENET_LABELS[top5_catid[j][k]]
                        if top5_catid[j][k] < len(IMAGENET_LABELS)
                        else "unknown"
                    )
                    preds.append({"label": label, "score": top5_prob[j][k].item()})
                predictions[filtered_indices[i + j]] = preds

    return predictions


def composition_runner(image, clf_model, config: ExperimentConfig):
    relevant_labels = {
        "2": "bicycle",
        "3": "car",
        "6": "bus",
        "7": "train",
        "8": "truck",
    }
    predictions = []

    upscaled_image = upscaling(image, config.gpu_id)

    detected_patches = segmentation(upscaled_image, relevant_labels, config.gpu_id)

    for patch, score, label, box in detected_patches:
        if not clf_model["is_timm"]:
            prediction = classification(patch, clf_model["model_id"], config.gpu_id)
        else:
            print(
                f"Model {clf_model['model_id']} depends on TIMM. Choosing TIMM pipeline."
            )
            prediction = timm_classification(
                patch, clf_model["model_id"], config.gpu_id
            )

        max_score_prediction = max(prediction, key=lambda x: x["score"])
        predictions.append(max_score_prediction)
    return predictions


def batch_processor(dataset, cls_model, config: ExperimentConfig):
    relevant_labels = {
        "2": "bicycle",
        "3": "car",
        "6": "bus",
        "7": "train",
        "8": "truck",
    }

    all_detected_patches = batch_segmentation(
        dataset["image"], relevant_labels, config.gpu_id, batch_size=64
    )

    detected_patches_with_ids = []

    for idx, patches in enumerate(all_detected_patches):
        detected_patches_with_ids.append((idx, patches))

    # print(
    #     detected_patches_with_ids,
    #     f"Length of all detected patches: {len(all_detected_patches)}",
    # )

    detected_images = []
    for _, patches in detected_patches_with_ids:
        for patch in patches:
            detected_images.append(patch[0])

    if not cls_model["is_timm"]:
        classification_results = batch_classification(
            detected_images, cls_model["model_id"], config.gpu_id, batch_size=128
        )

        classified_results_with_ids = []
        image_idx = 0
        for idx, patches in detected_patches_with_ids:
            image_predictions = []
            for _ in patches:
                if classification_results[image_idx] is not None:
                    image_predictions.append(classification_results[image_idx])
                image_idx += 1
            if image_predictions:
                highest_score_prediction = max(
                    image_predictions, key=lambda x: x[0]["score"]
                )
                classified_results_with_ids.append(highest_score_prediction)
            else:
                classified_results_with_ids.append([])
        return classified_results_with_ids
    else:
        classification_results = batch_timm_classification(
            detected_images, cls_model["model_id"], config.gpu_id, batch_size=128
        )

        classified_results_with_ids = []
        image_idx = 0
        for idx, patches in detected_patches_with_ids:
            image_predictions = []
            for _ in patches:
                if classification_results[image_idx] is not None:
                    image_predictions.append(classification_results[image_idx])
                image_idx += 1
            if image_predictions:
                highest_score_prediction = max(
                    image_predictions, key=lambda x: x[0]["score"]
                )
                classified_results_with_ids.append(highest_score_prediction)
            else:
                classified_results_with_ids.append([])
        return classified_results_with_ids


def ground_truth_runner(dataset, clf_models: CLS_Models, config: ExperimentConfig):
    model_count = 0
    models = []
    accuracies = []
    clf_models.random_sample_inference_models(
        len(clf_models.models["model_id"])
    )  # Sample all models as inference models.
    print("--------------------------------------------------")
    for index, model in clf_models.inference_models.iterrows():
        model_count += 1
        print(
            f"Model {model_count} out of {len(clf_models.inference_models['model_id'])}: {model['model_id']}"
        )

        preds = []
        true_labels = []
        if config.batch_processing == "True":
            preds = batch_processor(dataset, model, config)
            true_labels = dataset["label"]
        else:
            for data in dataset:
                composition_result = composition_runner(data["image"], model, config)
                preds.append(composition_result)
                true_labels.append(data["label"])

        accuracy = map_labels_for_accuracy(true_labels, preds, config.output_dir)
        print(f"Accuracy for {model['model_id']} is {accuracy}.")
        models.append(model["model_id"])
        accuracies.append(accuracy)
        print("--------------------------------------------------")

    results = [
        {
            "model_id": model,
            "inference_accuracy": acc,
        }
        for model, acc in zip(models, accuracies)
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{config.output_dir}/inference_results.csv")


def experiment_runner(dataset, clf_models: CLS_Models, config: ExperimentConfig):
    sampled_dataset = image_sampler(
        dataset,
        gpu_id=config.gpu_id,
        n_samples=config.n_samples,
        sampling_technique=config.sampling,
    )
    print(sampled_dataset)

    model_count = 0
    models = []
    accuracies = []
    cls_models.random_sample_inference_models(config.m_samples)

    print("--------------------------------------------------")
    for index, model in clf_models.inference_models.iterrows():
        model_count += 1
        print(
            f"Model {model_count} out of {len(clf_models.inference_models['model_id'])}: {model['model_id']}"
        )

        preds = []
        true_labels = []
        if config.batch_processing == "True":
            preds = batch_processor(dataset, model, config)
            true_labels = dataset["label"]
        else:
            for data in sampled_dataset:
                composition_result = composition_runner(data["image"], model, config)
                preds.append(composition_result)
                true_labels.append(data["label"])

        accuracy = map_labels_for_accuracy(true_labels, preds, config.output_dir)
        print(f"Accuracy for {model['model_id']} is {accuracy}.")
        models.append(model["model_id"])
        accuracies.append(accuracy)
        print("--------------------------------------------------")

    results = [
        {
            "model_id": model,
            "inference_accuracy": acc,
        }
        for model, acc in zip(models, accuracies)
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{config.output_dir}/inference_results.csv")
    clf_models.inference_models = pd.merge(
        clf_models.inference_models, results_df, on="model_id", how="left"
    )

    features, y = preprocessing(clf_models.inference_models)
    pipeline = configure_pipeline()

    cross_val_scores = cross_validation(
        pipeline, features, y, len(clf_models.inference_models["model_id"])
    )
    print(f"Cross validation score: {cross_val_scores}.")

    trained_pipeline = pipeline.fit(features, y)
    test_features, test_y = preprocessing(clf_models.test_models)

    print("--------------------------------------------------")

    clf_models.test_models["predicted_accuracy"] = None

    model_count = 0
    for index, model in clf_models.test_models.iterrows():
        model_count += 1
        model_id = model["model_id"]
        # print(
        #     f"Prediction for {model_id}. {model_count} out of {len(clf_models.test_models['model_id'])}."
        # )

        predicted_accuracy = predict_accuracy(
            trained_pipeline,
            test_features[test_features["model_id"] == model_id],
        )[0]

        # if model_id in test_features["model_id"].values:
        #     filtered_features = test_features.loc[test_features["model_id"] == model_id]
        #     predicted_accuracy = predict_accuracy(
        #         trained_pipeline,
        #         filtered_features,
        #     )[0]

        clf_models.test_models.at[index, "predicted_accuracy"] = predicted_accuracy

        print(f"Predicted accuracy values for {model_id}: {predicted_accuracy}")
        print("--------------------------------------------------")

    clf_models.test_models.to_csv(f"{config.output_dir}/test_results.csv")

    config.success = True
    config.save_experiment_config()


def image_sampler(dataset, gpu_id, n_samples, sampling_technique="random"):
    processor, model = init_feature_extractor(gpu_id)
    similarity_threshold = 0.1
    density_threshold = 35
    max_samples = 40
    embeddings_dict = {}
    batch_size = 128
    # density_scores = calculate_density(embeddings_dict)
    sampled_embeddings = {}
    sampled_indices = []

    for batch in batched_dataset(dataset, batch_size):
        result = predict_batch(processor, model, batch, gpu_id=gpu_id)
        embeddings_dict.update(result)

    if sampling_technique == "random":
        sampled_embeddings, sampled_indices = random_sampling(
            embeddings_dict, n_samples=n_samples
        )
    elif sampling_technique == "kmeans":
        sampled_embeddings, sampled_indices = kmeans_sampling(
            embeddings_dict, n_samples=n_samples
        )

    sampled_dataset = convert_to_hf_dataset(dataset, sampled_indices)

    return sampled_dataset


def get_classification_models_from_csv(csv_path):
    clf_models_df = pd.read_csv(csv_path)
    return clf_models_df


def create_dataset(filepath, n_samples=5849):
    dataset = load_dataset(
        "imagefolder",
        data_dir=filepath,
        split="test",
    )
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference tests.")
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use for computation."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5849,
        help="Number of images to be sampled for inference.",
    )
    parser.add_argument(
        "--m_samples",
        type=int,
        default=24,
        help="Number of models to be sampled for inference.",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of experiment runs."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Output directory for all the output files.",
        required=True,
    )
    parser.add_argument(
        "--desc",
        type=str,
        help="Description for this particular experiment run(s).",
        required=True,
    )
    parser.add_argument(
        "--is_gt",
        type=str,
        help="Whether the current run is for getting ground truth or not.",
        required=True,
    )
    parser.add_argument(
        "--clf_csv_path",
        type=str,
        default="/workspaces/ModelHubTest/src/composition/composition_classification_models.csv",
        help="Path to the csv file containing all the classification models.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspaces/ModelHubTest/src/dataset/vehicles",
        help="Path to the folder containing all the image files.",
    )
    parser.add_argument(
        "--batch_processing",
        type=str,
        default="False",
        help="Whether the object detection and image classification should be run in batches.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="random",
        help="Select the sampling technique used for sampling user input.",
    )
    args = parser.parse_args()
    return args


def evaluate_prediction_accuracy(config: ExperimentConfig):
    all_results = []
    ground_truth_df = pd.read_csv(
        "/workspaces/ModelHubTest/src/data/experiments/composition_experiments/may28-groundtruth/run_0/inference_results.csv"
    )
    for run in range(config.runs):
        result_file = f"{config.root_dir}/run_{run}/test_results.csv"
        if os.path.exists(result_file):
            result = pd.read_csv(result_file).drop("Unnamed: 0", axis=1)
            all_results.append(result)

    if not all_results:
        print("No results found.")
        return

    conc_results_df = pd.concat(all_results, ignore_index=True)

    aggregated_results_df = (
        conc_results_df.groupby("model_id")
        .agg(
            {
                "predicted_accuracy": "mean",
                "accuracy": "first",
                "dataset": "first",
                "base_model": "first",
                "likes": "first",
                "downloads": "first",
                "is_timm": "first",
            }
        )
        .reset_index()
    )

    merged_results_df = pd.merge(
        aggregated_results_df, ground_truth_df, on="model_id", how="inner"
    )

    # Calculate MAE, MSE, R² for each model
    merged_results_df["MAE"] = abs(
        merged_results_df["inference_accuracy"] - merged_results_df["predicted_accuracy"]
    )
    merged_results_df["MSE"] = (
        merged_results_df["inference_accuracy"] - merged_results_df["predicted_accuracy"]
    ) ** 2
    # merged_results_df["R2"] = merged_results_df.apply(
    #     lambda row: r2_score([row["inference_accuracy"]], [row["predicted_accuracy"]]),
    #     axis=1,
    # )
    mae = mean_absolute_error(merged_results_df['inference_accuracy'], merged_results_df['predicted_accuracy'])
    mse = mean_squared_error(merged_results_df['inference_accuracy'], merged_results_df['predicted_accuracy'])
    r2 = r2_score(merged_results_df['inference_accuracy'], merged_results_df['predicted_accuracy'])

    overall_metrics_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2],
        "sampling": [config.sampling],
        "n_samples": [config.n_samples],
        "m_samples": [config.m_samples],
        "runs": [config.runs],
    })
    overall_metrics_df.to_csv(f"{config.root_dir}/overall_metrics.csv", index=False)
    print(overall_metrics_df)

if __name__ == "__main__":
    args = parse_args()

    config = ExperimentConfig(
        args.runs,
        args.root_dir,
        args.desc,
        args.gpu_id,
        args.is_gt,
        args.clf_csv_path,
        args.dataset_path,
        args.n_samples,
        args.m_samples,
        args.batch_processing,
        args.sampling,
    )
    cls_models = CLS_Models(config.clf_csv_path)
    IMAGENET_LABELS = get_imagenet_labels()
    # cls_models.random_sample_inference_models(config.m_samples)
    config.root_dir = f"{config.root_dir}/experiments/{config.experiment_time}" if config.is_gt == "False" else f"{config.root_dir}/groundtruth/{config.experiment_time}"
    for run in range(config.runs):
        config.success = False
        config.output_dir = f"{config.root_dir}/run_{run}"
        create_directory(config.output_dir)
        if config.is_gt == "True":
            print(
                "Ground truth is on. Overriding user values for n_samples, m_samples."
            )
            config.m_samples = len(cls_models.models["model_id"])
            cls_models.random_sample_inference_models(config.m_samples)
            config.n_samples = 5849
            config.save_experiment_config()
            vehicle_dataset = create_dataset(config.dataset_path, config.n_samples)
            ground_truth_runner(vehicle_dataset, cls_models, config)
        else:
            print("Ground truth runner is off. Runnning experiment.")
            config.save_experiment_config()
            vehicle_dataset = create_dataset(config.dataset_path, config.n_samples)
            experiment_runner(vehicle_dataset, cls_models, config)

    evaluate_prediction_accuracy(config)
