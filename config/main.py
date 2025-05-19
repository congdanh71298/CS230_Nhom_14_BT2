config = {
    "convnext": {
        "batch_size": 56,
        "num_epochs": 6,
        "lr": 10**-4,
        "model_folder": "convnext",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning"
    },
    "swin": {
        "batch_size": 56,
        "num_epochs": 6,
        "lr": 10**-4,
        "model_folder": "swin",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning"
    },
    "vit": {
        "batch_size": 96,
        "num_epochs": 6,
        "lr": 10**-4,
        "model_folder": "vit",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "densenet": {
        "batch_size": 96,
        "num_epochs": 16,
        "lr": 10**-4,
        "model_folder": "densenet",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "efficientnet": {
        "batch_size": 164,
        "num_epochs": 31,
        "lr": 10**-4,
        "model_folder": "efficientnet",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "customcnn": {
        "batch_size": 64,
        "num_epochs": 30,
        "lr": 10**-3,
        "model_folder": "customcnn",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "resnet18": {
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 10**-4,
        "model_folder": "resnet18",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
    "vgg16": {
        "batch_size": 64,  # Smaller batch size due to larger model
        "num_epochs": 20,
        "lr": 10**-4,
        "model_folder": "vgg16",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
    },
}


def get_config(name):
    assert config.get(name) is not None, "Model not support yet"
    return config[name]
