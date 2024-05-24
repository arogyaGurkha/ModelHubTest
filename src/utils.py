from dotenv import dotenv_values
import pandas as pd
from datasets import load_dataset
from yaml import safe_load
from datetime import datetime
from pytz import timezone
import os

def get_hfapi_key(path):
    hf_token = dotenv_values(path)["HF_TOKEN"]
    return hf_token


def read_csv(path):
    dataframe = pd.read_csv(path)
    return dataframe

def create_cat_vs_dog_dataset(path):
    dataset = load_dataset("imagefolder", data_dir=path)
    return dataset

def parse_yaml(text):
    """
    Parse card_data to yaml.
    """
    return safe_load(str(text))

def get_current_time():
    utc_9 = timezone('Asia/Tokyo')
    utc_time = datetime.now().astimezone(utc_9)
    return f"{utc_time.hour}:{utc_time.minute}:{utc_time.second}"

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)