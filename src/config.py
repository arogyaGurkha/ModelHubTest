class Environment:
    ENV_FILE_PATH = "/workspaces/ModelHubTest/src/.env"
    ALL_MODEL_DATA_PATH = "/workspaces/ModelHubTest/src/data/experiments/n_image_inference/Cats_Vs_Dogs.csv"
    TOTAL_IMAGE_COUNT = 3618
    TOTAL_MODEL_COUNT = 52
    EXPERIMENT_MODEL_COUNT = 40
    GPU_ID = 0
    CLASSIFICATION_MODELS = {
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
        # "Sendeky/Cifar10",
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


class Settings:
    pass


class Experiment:
    def __init__(self, n_samples, m_samples, experiment_time, k_folds):
        """
        n_samples is the number of input samples that need inference.
        m_samples is the number of inference models.
        experiment_time is the timestamp or description of the experiment time
        """
        self.n_samples = n_samples
        self.m_samples = m_samples
        self.experiment_time = experiment_time
        self.k_folds = k_folds
        self.cross_val_scores = []
        self.inference_models = None
        self.test_models = None
        self.img_dataset = None
        self.inference_results = None
        self.prediction_data = None
        self.success = False
        # self.pytorch_profiler = None
        # self.inference_path = None
        # self.profiler_path = None

    def __str__(self):
        return (
            f"Experiment(n_samples={self.n_samples}, "
            f"m_samples={self.m_samples}, "
            f"experiment_time='{self.experiment_time}', "
            f"inference_models={self.inference_models}, "
            f"dataset={self.img_dataset})"
        )

    def get_experiment_config(self):
        config = {
            "success": self.success,
            "n_samples": self.n_samples,
            "m_samples": self.m_samples,
            "experiment_time": self.experiment_time,
            "k_folds": self.k_folds,
            "cross_val_score": list(self.cross_val_scores),
            "inference_models": self.inference_models,
            "test_models": self.test_models
        }
        return config

    def profiler_logs(self):
        pass
