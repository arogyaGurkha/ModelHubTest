"""
Run all models.
"""

from huggingface_model_server import HF_Models
from utils import get_hfapi_key, create_cat_vs_dog_dataset

import pandas as pd
import yaml
import transformers
from datasets import load_dataset, ClassLabel, Features, Image
from transformers.pipelines.pt_utils import KeyDataset
import cProfile
import pstats
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from config import Environment

DOTENV_PATH = "/workspaces/ModelHubTest/src/.env"

def run_baseline_method(models, dataset):
    # pr = cProfile.Profile()

    model_count = 0
    for model in models.keys():
        print(f"Running model: {model}. {model_count + 1} out of {len(models.keys())}.")
        file_name = model.split("/")[0] + "-" + model.split("/")[1]

        classifier = transformers.pipeline(model=model, device=0)
        predictions = []
        # pr.enable()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(
                dir_name=f"/workspaces/ModelHubTest/src/data/model_stats/log/{file_name}",
            ),
        ) as prof:
            for pred in classifier(KeyDataset(dataset["test"], "image"), batch_size=128):
                prof.step()
                predictions.append(pred)

        save_inference_predictions(dataset, model, predictions, file_name)
        # stats = pstats.Stats(pr).sort_stats("cumtime")
        # stats.strip_dirs()
        # stats.dump_stats(
        #     f"/workspaces/ModelHubTest/src/data/model_stats/cProfile_dump/{csv_name}.dmp"
        # )

        print("--------------------------------------------------")


def save_inference_predictions(dataset, model, preds, csv_name):
    predicted_labels = [pred[0]["label"] for pred in preds]
    labels = [item["label"] for item in dataset["test"]]

    results = [
        {
            "model_id": model,
            "true_label": true_label,
            "predicted_label": predicted_label,
        }
        for true_label, predicted_label in zip(labels, predicted_labels)
    ]
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"/workspaces/ModelHubTest/src/data/model_stats/inference_data/{csv_name}.csv"
    )


def sanity_test(dataset):
    classifier = transformers.pipeline(
        model="phuong-tk-nguyen/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
        device=0,
    )
    predictions = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(
            dir_name="/workspaces/ModelHubTest/src/data/model_stats/log/sanity",
        ),
    ) as prof:
        for pred in classifier(KeyDataset(dataset["test"], "image"), batch_size=128):
            prof.step()
            predictions.append(pred)
    print(predictions)

if __name__ == "__main__":
    # INITIALIZE
    hf = HF_Models(get_hfapi_key(DOTENV_PATH))
    models = Environment.CLASSIFICATION_MODELS
    valid_models = hf.download_models(models)
    dataset = load_dataset(
        "imagefolder",
        data_dir="/workspaces/ModelHubTest/src/data/assets/cat_vs_dog",
    )

    # sampled_models = hf.random_model_sample(10)
    # print(sampled_models.keys())

    # sanity_test(dataset)

    # df = pd.DataFrame(hf.models)
    # df.transpose().to_csv("/workspaces/ModelHubTest/src/data/Cats_Vs_Dogs.csv")
    # dataset = create_cat_vs_dog_dataset(
    #     "/workspaces/ModelHubTest/src/data/assets/cat_vs_dog"
    # )

    # models = ground_truth_models()
    # valid_models = download_models(hf, models)
    # features = Features({"image": Image(), "label": ClassLabel(num_classes=2, names=['cat', 'dog'])})

    # print(dataset["test"][0])

    # profiler = cProfile.Profile()
    # profiler.enable()
    # run_baseline_method(hf.models, dataset)
    # profiler.disable()
    # main_stats = pstats.Stats(profiler).sort_stats("cumtime")
    # main_stats.strip_dirs()
    # main_stats.print_stats()
    # main_stats.dump_stats("/workspaces/ModelHubTest/src/data/model_stats/baseline.dmp")

    # classifier = transformers.pipeline(
    #     model="phuong-tk-nguyen/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
    #     device=0,
    # )

    # pr = cProfile.Profile()
    # pr.enable()
    # count = 0
    # for pred in classifier(KeyDataset(dataset["test"], "image"), batch_size=1):
    #     print(pred)

    # print(dataset['test']['label'])

    # pr.disable()
    # stats = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # stats.strip_dirs()
    # stats.print_stats()
