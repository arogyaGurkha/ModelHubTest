from dotenv import dotenv_values
import pandas as pd
from datasets import load_dataset


def get_hfapi_key(path):
    hf_token = dotenv_values(path)["HF_TOKEN"]
    return hf_token


def read_csv(path):
    dataframe = pd.read_csv(path)
    return dataframe

def create_cat_vs_dog_dataset(path):
    dataset = load_dataset("imagefolder", data_dir=path)
    return dataset
