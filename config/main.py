config = {
    "convnext": {
        "batch_size": 56,
        "num_epochs": 5,
        "lr": 10**-4,
        "model_folder": "convnext",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
        "drive": "https://drive.google.com/file/d/1KCo-Wi99WyXQsil-bGHe2jfUURJxNv63/view?usp=sharing"
    },
    "swin": {
        "batch_size": 56,
        "num_epochs": 5,
        "lr": 10**-4,
        "model_folder": "swin",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
        "drive": "https://drive.google.com/file/d/1yN30YDnqJTV2vyuZlIxRR0j8c_RJEg-4/view?usp=sharing"
    },
    "vit": {
        "batch_size": 96,
        "num_epochs": 5,
        "lr": 10**-4,
        "model_folder": "vit",
        "preload": "latest",
        "ds_width": 224,
        "ds_height": 224,
        "model_basename": "finetuning",
        "drive": "https://drive.google.com/file/d/1G7if7HOoIoxQoXIgLh1ClsBWBGLMb1RC/view?usp=drive_link"
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
        "batch_size": 164,
        "num_epochs": 35,
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
