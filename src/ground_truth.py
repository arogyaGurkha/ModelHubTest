"""
Run all models.
"""

from huggingface_model_server import HF_Models
from utils import get_hfapi_key, create_cat_vs_dog_dataset

import pandas as pd
import yaml

DOTENV_PATH = "/workspaces/ModelHubTest/src/.env"


def ground_truth_models():
    classification_models = {
        # "google/vit-base-patch16-224",
        # "microsoft/resnet-50",
        # "microsoft/beit-base-patch16-224-pt22k-ft22k",
        # "apple/AIM",
        # "microsoft/resnet-18",
        # "apple/mobilevit-small",
        # "timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k",
        # "microsoft/swin-tiny-patch4-window7-224",
        # "nvidia/mit-b0",
        # "timm/mobilenetv3_large_100.ra_in1k",
        "akahana/vit-base-cats-vs-dogs",
        "ismgar01/vit-base-cats-vs-dogs",
        "nateraw/vit-base-cats-vs-dogs",
        "efederici/convnext-base-224-22k-1k-orig-cats-vs-dogs",
        "tangocrazyguy/resnet-50-finetuned-cats_vs_dogs",
        "danieltur/my_awesome_catdog_model",
        "Natalia2314/vit-base-catsVSdogs-demo-v5",
        "Camilosan/Modelo-catsVSdogs",
        "Dricz/cat-vs-dog-resnet-50",
        "mhdiqbalpradipta/cat_or_dogs",
        "Amadeus99/cat_vs_dog_classifier",
        "cppgohan/resnet-50-finetuned-dog-vs-cat",
        "02shanky/vit-finetuned-vanilla-cifar10-0",
        "jadohu/BEiT-finetuned",
        "jimypbr/cifar10_outputs",
        "DunnBC22/vit-base-patch16-224-in21k_dog_vs_cat_image_classification",
        "heyitskim1912/tknnguyen.2022_AML_A2_Q4",
        "Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
        "Weili/vit-base-patch16-224-finetuned-cifar10",
        "tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        "phuong-tk-nguyen/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
        "phuong-tk-nguyen/vit-base-patch16-224-finetuned-cifar10",
        "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "againeureka/vit_cifar10_classification_tmp",
        "xlagor/swin-tiny-patch4-window7-224-finetuned-fit",
        "Skafu/swin-tiny-patch4-window7-224-cifar10",
        "tejp/finetuned-cifar10",
        "jcollado/swin-tiny-patch4-window7-224-finetuned-cifar10",
        "jagriti/swin-tiny-patch4-window7-224",
        "phuong-tk-nguyen/vit-base-patch16-224-finetuned",
        "rs127/swin-tiny-patch4-window7-224-finetuned-cifar10",
        "TirathP/cifar10-lt",
        "phuong-tk-nguyen/vit-base-patch16-224-newly-trained",
        "ahirtonlopes/swin-tiny-patch4-window7-224-finetuned-cifar10",
        "phuong-tk-nguyen/swin-base-patch4-window7-224-in22k-newly-trained",
        "vhurryharry/cifarv2",
        "Skafu/swin-tiny-patch4-window7-224-finetuned-eurosat-finetuned-cifar100",
        "vhurryharry/cifar",
        "vhurryharry/cifarv2",
        "phuong-tk-nguyen/None",
        "phuong-tk-nguyen/swin-tiny-patch4-window7-224-finetuned",
        "phuong-tk-nguyen/swin-tiny-patch4-window7-224-finetuned-eurosat",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.3-finetuned-eurosat",
        "jorgeduardo13/mobilevit-small-finetuned",
        "Sendeky/Cifar10",
        "arize-ai/resnet-50-cifar10-quality-drift",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.4-finetuned-eurosat",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.5-finetuned-eurosat",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.1-Ln-finetuned-eurosat",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.6-finetuned-eurosat",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.2-finetuned-eurosat",
        "2022happy/swin-tiny-patch4-window7-224-pruned-0.1-finetuned-eurosat",
        "simlaharma/vit-base-cifar10",
        "karthiksv/vit-base-patch16-224-cifar10",
        "karthiksv/vit-base-patch16-224-in21k-finetuned-cifar10",
    }
    return classification_models


def download_models(hf: HF_Models, models):
    for model in models:
        model_info = [m for m in hf.list_models(model)]
        for _model in model_info:
            if _model.card_data:
                card_data = yaml.safe_load(str(_model.card_data))
                if is_model_valid(card_data):
                    hf.add_model(
                        {
                            "model": _model.id,
                            "accuracy": hf.extract_accuracy(card_data),
                            "dataset": hf.extract_dataset(_model.id, card_data),
                            "base_model": hf.extract_base_model_id(
                                _model.id, card_data
                            ),
                            "likes": _model.likes,
                            "downloads": _model.downloads,
                        }
                    )
                else:
                    print(f"{_model.id} discarded.")
    print(f"{len(hf.models.keys())} out of {len(models)} saved.")


def is_model_valid(card_data):
    necessary_keys = {"task", "dataset", "metrics"}
    return any(
        necessary_keys.issubset(result)
        for entry in card_data.get("model-index", [])
        for result in entry.get("results", [])
    )


def hf_modelinfo_download(hf: HF_Models):
    models = hf.hf_api.list_models(trained_dataset="cats_vs_dogs", cardData=True)
    for model in models:
        if model.card_data:
            card_data = yaml.safe_load(str(model.card_data))
            if is_model_valid(card_data):
                hf.add_model(
                    {
                        "model": model.id,
                        "accuracy": hf.extract_accuracy(card_data),
                        "dataset": hf.extract_dataset(model.id, card_data),
                        "base_model": hf.extract_base_model_id(model.id, card_data),
                        "likes": model.likes,
                        "downloads": model.downloads,
                    }
                )


if __name__ == "__main__":
    hf = HF_Models(get_hfapi_key(DOTENV_PATH))
    # models = ground_truth_models()
    # valid_models = download_models(hf, models)
    # df = pd.DataFrame(hf.models)
    # df.transpose().to_csv("/workspaces/ModelHubTest/src/data/Cats_Vs_Dogs.csv")
    dataset = create_cat_vs_dog_dataset("/workspaces/ModelHubTest/src/data/assets/cat_vs_dog")
    print(dataset)
    # print(hf.cache)
