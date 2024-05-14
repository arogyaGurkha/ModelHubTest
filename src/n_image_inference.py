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

ENV_FILE_PATH = "/workspaces/ModelHubTest/src/.env"


def create_dataset(n_samples):
    dataset = load_dataset(
        "imagefolder",
        data_dir="/workspaces/ModelHubTest/src/data/assets/cat_vs_dog",
        split="test",
    )
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset
    # dataset = dataset.select(range(n_samples))


def get_models(hf: HF_Models, m_samples):
    all_models = Environment.CLASSIFICATION_MODELS
    hf.download_models(all_models)
    samples = hf.random_model_sample(m_samples)
    print(f"Sampled {m_samples} models out of {len(all_models)}")
    return samples


def calculate_performance_metrics(labels, preds):
    print("Calculating performance metrics...")

    #FIXME: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
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
            dir_name=f"/workspaces/ModelHubTest/src/data/model_stats/log/{exp_time}/{model_id.replace('/', '-')}",
        ),
    ) as prof:
        for pred in classifier(KeyDataset(dataset, "image"), batch_size=128):
            prof.step()
            predictions.append(pred)
    return predictions


def run_experiment(exp_config: Experiment):
    model_count = 0
    experiment_data_directory = f"/workspaces/ModelHubTest/src/data/model_stats/inference_data/{exp_config.experiment_time}"
    create_directory(experiment_data_directory)

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
            f"{experiment_data_directory}/{model_id.replace('/', '-')}.csv"
        )

        # TODO: Mapping for labels, so evaluate can calculate them
        label_map = {"cat": 0, "dog": 1}
        def label_map_with_ignore(label):
            return label_map.get(label, -1)

        metrics = calculate_performance_metrics(
            inference_results_df["true_label"].map(label_map_with_ignore).to_list(),
            inference_results_df["predicted_label"]
            .str.lower()
            .map(label_map_with_ignore)
            .to_list(),
        )

        exp_config.inference_results[model_id.id].update(metrics)
        print(exp_config.inference_results)

        print("--------------------------------------------------")

        # cat_dog_models["pipeline_precision"] = cat_dog_models["model"].apply(
        #     lambda id: results[id]["precision"] if id in results else -1
        # )
    # Calculate Model Performance
    # Create Dataset
    # Perform k-fold training
    # Testing


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
        n_samples=2000, m_samples=40, experiment_time=get_current_time()
    )
    hf = HF_Models(get_hfapi_key(ENV_FILE_PATH))

    experiment_config.inference_models = get_models(hf, experiment_config.m_samples)
    experiment_config.img_dataset = create_dataset(experiment_config.n_samples)

    # print(experiment_config.inference_models)

    run_experiment(experiment_config)
    # run_inference(e_b, e_b.inference_models, n_image_samples)
    # print(experiment)
