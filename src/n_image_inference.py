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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MultiLabelBinarizer,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm
from termcolor import colored


def create_dataset(n_samples=3618):
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
    print(f"Sampled {m_samples} models out of {Environment.TOTAL_MODEL_COUNT}")
    return samples


def get_test_models(hf: HF_Models):
    remaining_models = hf.models.keys()


def calculate_performance_metrics(labels, preds):
    print("Calculating performance metrics...")

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    result = clf_metrics.compute(references=labels, predictions=preds)
    result = {f"pred_{key}": value for key, value in result.items()}
    print(f"Performance metrics: {result}")

    return result


def run_single_inference(exp_time, model_id, dataset):
    print(colored(f"Running model: {model_id}.", "yellow"))

    classifier = transformers.pipeline(model=model_id, device=Environment.GPU_ID)
    predictions = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(
            dir_name=f"/workspaces/ModelHubTest/src/data/experiments/n_image_inference/{exp_time}/log/{model_id.replace('/', '-')}",
        ),
    ) as prof:
        for pred in tqdm(
            classifier(KeyDataset(dataset, "image"), batch_size=128), total=len(dataset)
        ):
            prof.step()
            predictions.append(pred)
    return predictions


def map_labels(labels, preds):
    label_map = {"cat": 0, "dog": 1, "cats": 0, "dogs": 1}

    def label_map_with_toggle(label, pred):
        if pred in label_map:
            return label_map[pred]
        else:
            return 1 - label

    mapped_preds = [label_map_with_toggle(t, p) for t, p in zip(labels, preds)]

    return mapped_preds


def all_model_data():
    return pd.read_csv(Environment.ALL_MODEL_DATA_PATH)


def preprocessing(data):
    print("Preprocessing dataset...")
    data["dataset"] = data["dataset"].apply(lambda x: ",".join(sorted(x)))

    if "pred_precision" in data.columns:
        features = data.drop("pred_precision", axis=1)
        label = data["pred_precision"]
    else:
        # For new data with no labels (pred_precision)
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


def predict_precision(pipeline, new_data):
    new_preds = pipeline.predict(new_data)
    return new_preds


def run_experiment(exp_config: Experiment):
    experiment_data_directory = f"/workspaces/ModelHubTest/src/data/experiments/n_image_inference/{exp_config.experiment_time}/"
    create_directory(f"{experiment_data_directory}/inference_data/")

    save_experiment_config(
        exp_config.get_experiment_config(),
        f"{experiment_data_directory}/experiment_config.json",
    )

    exp_config.inference_results = exp_config.inference_models
    model_count = 0
    print("--------------------------------------------------")
    for model_id in exp_config.inference_models.keys():
        model_count += 1
        print(
            colored(
                f"Model {model_count} out of {len(exp_config.inference_models.keys())}.",
                "yellow",
            )
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

        exp_config.inference_results[model_id].update(
            {"pred_precision": metrics["pred_precision"]}
        )

        print("--------------------------------------------------")

    results_df = pd.DataFrame(list(exp_config.inference_results.values()))
    results_df.to_csv(f"{experiment_data_directory}/inference_results.csv")

    features, y = preprocessing(results_df)
    pipeline = configure_pipeline()

    exp_config.cross_val_scores = cross_validation(
        pipeline, features, y, exp_config.k_folds
    )
    print(f"Cross validation score: {exp_config.cross_val_scores}.")

    # Testing

    trained_pipeline = pipeline.fit(features, y)

    test_models_df = (
        pd.DataFrame.from_dict(exp_config.test_models, orient="index")
        .reset_index()
        .drop(["index"], axis=1)
    )
    test_features, test_y = preprocessing(test_models_df)

    print("--------------------------------------------------")
    for model_id in exp_config.test_models.keys():
        model_count += 1
        # print(
        #     f"Prediction for {model_id}. {model_count} out of {len(exp_config.test_models.keys())}."
        # )

        predicted_precision = predict_precision(
            trained_pipeline, test_features[test_features["model"] == model_id]
        )[0]

        exp_config.test_models[model_id].update(
            {"predicted_precision": predicted_precision}
        )

        print(f"Predicted precision values for {model_id}: {predicted_precision}")
        print("--------------------------------------------------")

    test_df = pd.DataFrame(list(exp_config.test_models.values()))
    test_df.to_csv(f"{experiment_data_directory}/test_results.csv")

    exp_config.success = True
    save_experiment_config(
        exp_config.get_experiment_config(),
        f"{experiment_data_directory}/experiment_config.json",
    )


def save_experiment_config(config, filename):
    try:
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
        print(colored(f"Experiment configuration saved successfully to {filename}", "green"))
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
    hf = HF_Models(get_hfapi_key(Environment.ENV_FILE_PATH))
    downloaded_models = get_models(hf, Environment.EXPERIMENT_MODEL_COUNT)

    # image_counts = [200, 100, 50]
    image_counts = [Environment.TOTAL_IMAGE_COUNT, 3000, 2500, 2000, 1500, 1000, 500]
    for img_count in image_counts:
        print("--------------------------------------------------")
        print(colored(f"Running experiment on {img_count} images.", "magenta"))
        experiment_config = Experiment(
            n_samples=img_count,
            m_samples=Environment.EXPERIMENT_MODEL_COUNT,
            experiment_time=get_current_time(),
            k_folds=Environment.EXPERIMENT_MODEL_COUNT,
        )
        experiment_config.inference_models = downloaded_models
        experiment_config.test_models = {
            key: hf.models[key]
            for key in (hf.models.keys() - experiment_config.inference_models.keys())
        }
        experiment_config.img_dataset = create_dataset(experiment_config.n_samples)
        run_experiment(experiment_config)

    # ("datasets", MultiLabelBinarizerWrapper(), ["dataset"]),
    # ("base_model", OneHotEncoder(), ["base_model"]),
    # ("scaler", StandardScaler(), ["likes", "downloads"]),

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", StandardScaler(), ["accuracy", "likes", "downloads"]),
    #         ("datasets", MultiLabelBinarizerWrapper(), ["dataset"]),
    #     ],
    #     remainder="passthrough",
    # )
    # pipeline = Pipeline(
    #     steps=[
    #         ("preprocessor", preprocessor),
    #         ("model", RandomForestRegressor(random_state=42)),
    #     ]
    # )

    # print(pipeline)

    # pipeline.fit(
    #     data.drop(["pred_precision", "model"], axis=1),
    #     data["pred_precision"],
    # )

    # ("datasets", MultiLabelBinarizerWrapper(), ["dataset"]),
    # ("base_model", OneHotEncoder(), ["base_model"]),
    # ("scaler", StandardScaler(), ["likes", "downloads"]),

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", StandardScaler(), ["accuracy", "likes", "downloads"]),
    #         ("datasets", MultiLabelBinarizerWrapper(), ["dataset"]),
    #     ],
    #     remainder="passthrough",
    # )
    # pipeline = Pipeline(
    #     steps=[
    #         ("preprocessor", preprocessor),
    #         ("model", RandomForestRegressor(random_state=42)),
    #     ]
    # )

    # print(pipeline)

    # pipeline.fit(
    #     data.drop(["pred_precision", "model"], axis=1),
    #     data["pred_precision"],
    # )

    # experiment_config.inference_models = get_models(hf, experiment_config.m_samples)
    # experiment_config.img_dataset = create_dataset(experiment_config.n_samples)

    # data = {
    #     "accuracy": [0.9548, 0.967],
    #     "dataset": [["cifar10"], ["imagefolder", "imagenet-1k", "imagenet-21k"]],
    #     "base_model": [
    #         "microsoft/swin-tiny-patch4-window7-224",
    #         "google/vit-base-patch16-224",
    #     ],
    #     "likes": [0, 0],
    #     "downloads": [12, 7],
    #     "pred_precision": [0.0, 0.22],
    # }
    # df = pd.DataFrame(data)

    # df["dataset"] = df["dataset"].apply(lambda x: ",".join(sorted(x)))

    # X = df.drop("pred_precision", axis=1)
    # y = df["pred_precision"]

    # numeric_features = ["accuracy", "likes", "downloads"]
    # numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # categorical_features = ["base_model", "dataset"]
    # categorical_transformer = Pipeline(
    #     steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    # )

    # # print(categorical_transformer.fit(df['base_model']))

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", numeric_transformer, numeric_features),
    #         ("cat", categorical_transformer, categorical_features),
    #     ],
    #     remainder="drop",
    # )

    # pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    # X_preprocessed = pipeline.fit_transform(X)

    # print(X_preprocessed)
