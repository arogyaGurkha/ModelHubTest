from huggingface_hub import HfApi, ModelCard, utils
import re


class HF_Models:
    BASE_MODEL_PATTERN = re.compile(r"fine-tuned version of \[(.*?)\]")

    def __init__(self, hf_token):
        self.models = {}
        self.hf_api = self._configure_hfapi(hf_token)
        self.cache = {}
        self.card_data = {}

    def _configure_hfapi(self, hf_token):
        return HfApi(
            endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
            token=hf_token,  # Token is not persisted on the machine.
        )

    def load_model_info(self, model_id) -> ModelCard:
        if model_id not in self.cache:
            try:
                self.cache[model_id] = ModelCard.load(
                    model_id, ignore_metadata_errors=True
                )
            except utils.RepositoryNotFoundError:
                print("Repository not found for: " + model_id)
                self.cache[model_id] = None
            except utils.EntryNotFoundError:
                print("README not found for: " + model_id)
                self.cache[model_id] = None
            except IsADirectoryError:
                print(f"{model_id} is a directory?")
                self.cache[model_id] = None
        return self.cache[model_id]

    def get_base_model_dataset_from_hf(self, model_id):
        """Retrieve dataset information from HF for a base model."""
        if model_id:
            info = self.load_model_info(model_id)
            if info:
                return info.data.get("datasets")
        return None

    def extract_base_model_id(self, model_id, card_data):
        """Extract base model info"""
        base_model = card_data.get("base_model")
        if not base_model:
            info = self.load_model_info(model_id)
            if info:
                match = self.BASE_MODEL_PATTERN.search(info.text)
                base_model = match.group(1) if match else None
            else:
                base_model = None
        return base_model

    def extract_base_model_datasets(self, model_id, card_data):
        """Extract datasets for the base model."""
        base_model = self.extract_base_model_id(model_id, card_data)
        datasets = self.get_base_model_dataset_from_hf(base_model)
        print(f"Base model datasets for {model_id}: {datasets}")
        return datasets

    def add_model(self, model_info):
        self.models[model_info["model"]] = model_info

    def extract_dataset(self, model_id, card_data):
        def get_dataset_names(card_data):
            return [
                result.get("dataset", {}).get("name")
                for entry in card_data.get("model-index", [])
                for result in entry.get("results", [])
                if result.get("dataset", {}).get("name")
            ]

        datasets = get_dataset_names(card_data)
        if "imagefolder" in datasets or "image_folder" in datasets:
            base_model = self.extract_base_model_id(model_id, card_data)
            base_datasets = (
                self.extract_base_model_datasets(base_model, card_data) or []
            )
            return [datasets[0], *base_datasets]

        return datasets

    def extract_accuracy(self, card_data):
        metrics = (
            metric
            for entry in card_data.get("model-index", [])
            for result in entry.get("results", [])
            for metric in result.get("metrics", [])
        )

        for metric in metrics:
            if metric.get("type") == "accuracy":
                accuracy_value = metric.get("value")
                if isinstance(accuracy_value, list) and accuracy_value:
                    return float(accuracy_value[0])
                elif isinstance(accuracy_value, (float, int)):
                    return float(accuracy_value)

        return 0.0

    def list_models(self, model_id):
        return self.hf_api.list_models(model_name=model_id, cardData=True)
