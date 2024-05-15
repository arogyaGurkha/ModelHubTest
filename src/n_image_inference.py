from datasets import load_dataset
from config import Experiment, Environment
from huggingface_model_server import HF_Models
from utils import get_hfapi_key, get_current_time, create_directory
import transformers
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
import evaluate
import json

ENV_FILE_PATH = "/workspaces/ModelHubTest/src/.env"


def create_dataset(n_samples):
    dataset = load_dataset(
        "imagefolder",
        data_dir="/workspaces/ModelHubTest/src/data/assets/cat_vs_dog",
        split="test",
    )
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset


def get_models(hf: HF_Models, m_samples):
    all_models = Environment.CLASSIFICATION_MODELS
    hf.download_models(all_models)
    samples = hf.random_model_sample(m_samples)
    print(f"Sampled {m_samples} models out of {len(all_models)}")
    return samples


def calculate_performance_metrics(labels, preds):
    print("Calculating performance metrics...")

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    result = clf_metrics.compute(references=labels, predictions=preds)
    print(f"Performance metrics: {result}")
    return result


def run_single_inference(exp_time, model_id, dataset):
    print(f"Running model: {model_id}.")

    classifier = transformers.pipeline(model=model_id, device=0)
    predictions = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(
            dir_name=f"/workspaces/ModelHubTest/src/data/experiments/n_image_inference/{exp_time}/log/{model_id.replace('/', '-')}",
        ),
    ) as prof:
        for pred in classifier(KeyDataset(dataset, "image"), batch_size=128):
            prof.step()
            predictions.append(pred)
    return predictions


def map_labels(labels, preds):
    label_map = {"cat": 0, "dog": 1}

    def label_map_with_toggle(label, pred):
        if pred in label_map:
            return label_map[pred]
        else:
            return 1 - label

    mapped_preds = [label_map_with_toggle(t, p) for t, p in zip(labels, preds)]

    return mapped_preds


def run_experiment(exp_config: Experiment):
    experiment_data_directory = f"/workspaces/ModelHubTest/src/data/experiments/n_image_inference/{exp_config.experiment_time}/"
    create_directory(f"{experiment_data_directory}/inference_data/")
    save_experiment_config(
        exp_config.get_experiment_config(),
        f"{experiment_data_directory}/experiment_config.json",
    )

    exp_config.inference_results = exp_config.inference_models
    model_count = 0
    for model_id in exp_config.inference_models.keys():
        print(
            f"Model {model_count + 1} out of {len(exp_config.inference_models.keys())}."
        )

        inference_results = run_single_inference(
            exp_config.experiment_time, model_id, exp_config.img_dataset
        )
        inference_results_df = manage_inference_results(
            model_id, inference_results, exp_config.img_dataset
        )
        inference_results_df.to_csv(
            f"{experiment_data_directory}/inference_data/{model_id.replace('/', '-')}.csv"
        )

        inference_results_df["predicted_label"] = map_labels(
            inference_results_df["true_label"].to_list(),
            inference_results_df["predicted_label"].str.lower().to_list(),
        )

        metrics = calculate_performance_metrics(
            inference_results_df["true_label"].to_list(),
            inference_results_df["predicted_label"].to_list(),
        )

        exp_config.inference_results[model_id].update(metrics)

        print("--------------------------------------------------")

    results_df = pd.DataFrame(list(experiment_config.inference_results.values()))
    create_directory(f"{experiment_data_directory}/overall_results/")
    results_df.to_csv(
        f"{experiment_data_directory}/overall_results/{model_id.replace('/', '-')}.csv"
    )

    # Perform k-fold training
    # Testing


def save_experiment_config(config, filename):
    try:
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Experiment configuration saved successfully to {filename}")
    except IOError as e:
        print(f"Failed to write to {filename}: {e}")


def manage_inference_results(model, preds, dataset) -> pd.DataFrame:
    predicted_labels = [pred[0]["label"] for pred in preds]
    labels = [item["label"] for item in dataset]

    results = [
        {
            "model_id": model,
            "true_label": true_label,
            "predicted_label": predicted_label,
        }
        for true_label, predicted_label in zip(labels, predicted_labels)
    ]
    results_df = pd.DataFrame(results)

    return results_df


if __name__ == "__main__":
    experiment_config = Experiment(
        n_samples=2000, m_samples=2, experiment_time=get_current_time()
    )
    hf = HF_Models(get_hfapi_key(ENV_FILE_PATH))

    experiment_config.inference_models = get_models(hf, experiment_config.m_samples)
    experiment_config.img_dataset = create_dataset(experiment_config.n_samples)

    run_experiment(experiment_config)
