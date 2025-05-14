config = {
    "convnext": {
        "batch_size": 56,
        "num_epochs": 15,
        "lr": 10**-4,
        "model_folder": "convnext",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "swin": {
        "batch_size": 56,
        "num_epochs": 15,
        "lr": 10**-4,
        "model_folder": "swin",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "vit": {
        "batch_size": 96,
        "num_epochs": 15,
        "lr": 10**-4,
        "model_folder": "vit",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "densenet": {
        "batch_size": 96,
        "num_epochs": 15,
        "lr": 10**-4,
        "model_folder": "densenet",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "efficientnet": {
        "batch_size": 96,
        "num_epochs": 15,
        "lr": 10**-4,
        "model_folder": "efficientnet",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
}


def get_config(name):
    assert config.get(name) is not None, "Model not support yet"
    return config[name]
