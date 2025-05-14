from pathlib import Path


def get_config():
    return {
        "batch_size": 96,
        "num_epochs": 15,
        "lr": 10**-4,
        "model_folder": "ViT",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning"
    }
