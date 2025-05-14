import torch
from transformers import (
    ViTForImageClassification,
    SwinForImageClassification,
    ConvNextForImageClassification,
)
from torchvision.models import densenet121, efficientnet_b0
import torch.nn as nn
import argparse

from config.main import get_config
from utils.fine_tuning import finetuning


def get_ViT_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=100
    )
    return model


def get_DenseNet_model():
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 100)
    return model


def get_ConvNext_model():
    model = ConvNextForImageClassification.from_pretrained(
        "facebook/convnext-base-224",  # or 'convnext-small', 'convnext-tiny', etc.
        num_labels=100,
        ignore_mismatched_sizes=True,
    )
    return model


def get_Swin_model():
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224-in22k",
        num_labels=100,
        ignore_mismatched_sizes=True,
    )
    return model


def get_EfficientNet_model():
    model = efficientnet_b0(pretrained=True)  # Or b1, b2, ..., b7
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
    return model


def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "vit":
        return get_ViT_model()
    elif model_name == "swin":
        return get_Swin_model()
    elif model_name == "convnext":
        return get_ConvNext_model()
    elif model_name == "densenet":
        return get_DenseNet_model()
    elif model_name == "efficientnet":
        return get_EfficientNet_model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to load: vit, swin, convnext, densenet, efficientnet",
    )
    args = parser.parse_args()

    model = get_model(args.model)
    print(f"Loaded model: {args.model}")

    config = get_config(args.model)

    finetuning(config, model)
