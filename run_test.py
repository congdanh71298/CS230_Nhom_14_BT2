import argparse
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import matplotlib.pyplot as plt
from config.main import get_config
from dataset.cifar100 import get_cifar100_ds
from utils.get_model import get_model
from utils.fine_tuning import finetuning, cal_test_metrics
from utils.weight_retrieve import get_weights_file_path, latest_weights_file_path
import pandas as pd

if __name__ == "__main__":
    models = ["convnext", "vit", "swin", "efficientnet", "densenet"]
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
            raise Exception("No model to validate")

        print("-" * 50)
        print(f"Pretrain Model: {model_name};")

        # Get predictions and labels
        _, _, _, _, all_preds, all_labels = cal_test_metrics(
            model,
            val_data_loader,
            device,
            lambda msg: print(msg),
        )

        # Compute and save confusion matrix
        # cm = confusion_matrix(all_labels, all_preds, normalize="true")
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # fig, ax = plt.subplots(figsize=(15, 15))
        # disp.plot(cmap="Blues", ax=ax)
        # plt.title(f"Confusion Matrix - {model_name}")
        # plt.grid(False)
        # save_path = f"confusion_matrix_{model_name}.png"
        # plt.savefig(save_path, bbox_inches="tight")
        # plt.close(fig)

        # print(f"Saved confusion matrix: {save_path}")
        csv_path = f"predictions_{model_name}.csv"
        df = pd.DataFrame({"true_label": all_labels, "predicted_label": all_preds})
        df.to_csv(csv_path, index=False)
        print(f"Saved predictions to: {csv_path}")
        print("-" * 50)
