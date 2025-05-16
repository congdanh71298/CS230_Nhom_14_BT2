import argparse
import torch

from config.main import get_config
from dataset.cifar100 import get_cifar100_ds
from utils.get_model import get_model
from utils.fine_tuning import finetuning, run_validation
from utils.weight_retrieve import get_weights_file_path, latest_weights_file_path

if __name__ == "__main__":
    models = ['convnext', 'vit', 'swin', 'efficientnet', 'densenet']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in models:
        config = get_config(model_name)
        preload = config["preload"]
        model_filename = (
            latest_weights_file_path(config)
            if preload == "latest"
            else get_weights_file_path(config, preload) if preload else None
        )
        (train_data_loader, val_data_loader) = get_cifar100_ds(config)

        model = get_model(model_name)

        model.to(device)
        if model_filename:
            state = torch.load(model_filename)
            model.load_state_dict(state["model_state_dict"])
        else:
            raise Exception('No model to validate')

        val_accuracy = run_validation(
            model,
            val_data_loader,
            device,
            lambda msg: print(msg),
        )

        print(
            f'Pretrain Model: {model_name}; validation accuracy on cifar100: {val_accuracy}')
