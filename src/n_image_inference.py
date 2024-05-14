from datasets import load_dataset
from config import Experiment, Environment
from huggingface_model_server import HF_Models
from utils import get_hfapi_key

ENV_FILE_PATH = "/workspaces/ModelHubTest/src/.env"


def create_dataset(n_samples):
    dataset = load_dataset(
        "imagefolder",
        data_dir="/workspaces/ModelHubTest/src/data/assets/cat_vs_dog",
    )
    dataset = dataset.shuffle(seed=42).select(n_samples)


def get_models(hf: HF_Models, m_samples):
    hf.random_model_sample(m_samples)

if __name__ == "__main__":
    experiment = Experiment(n_samples=2000, m_samples=40)
    hf = HF_Models(get_hfapi_key(ENV_FILE_PATH))

    inference_models = get_models(hf, experiment.m_samples)
    n_image_samples = create_dataset(experiment.n_samples)
