import torch
from transformers import (
    ViTForImageClassification,
    SwinForImageClassification,
    ConvNextForImageClassification,
)
from torchvision.models import densenet121, efficientnet_b0, resnet18, vgg16
import torch.nn as nn
from types import SimpleNamespace


class HuggingFaceStyleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return SimpleNamespace(logits=logits)  # mimic HuggingFace output


def get_ViT_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=100
    )
    return model


def get_DenseNet_model():
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 100)
    return HuggingFaceStyleWrapper(model)


def get_ConvNext_model():
    model = ConvNextForImageClassification.from_pretrained(
        # or 'convnext-small', 'convnext-tiny', etc.
        "facebook/convnext-base-224",
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
    return HuggingFaceStyleWrapper(model)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Increased filters from 16 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Increased filters from 32 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Increased filters from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Increased filters from 128 to 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )
        
        # Fifth convolutional block
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Increased filters from 256 to 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )

        # Sixth convolutional block (New)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7 -> 3x3 (assuming input 224, output becomes 3x3)
        )
        
        # Fully connected layers
        # The input size to the linear layer depends on the output of the last pooling layer.
        # If input is 224x224, after 6 pooling layers (224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3), it's 3x3.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024), # Adjusted for new conv6 output and increased dense layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.4), # Keeping the modified dropout
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x) # Added conv6
        logits = self.classifier(x)
        return logits
        
def get_CustomCNN_model():
    model = CustomCNN(num_classes=100)
    return HuggingFaceStyleWrapper(model)


def get_ResNet18_model():
    model = resnet18(pretrained=True)
    # Adjust the final fully connected layer for CIFAR-100 (100 classes)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return HuggingFaceStyleWrapper(model)


def get_VGG16_model():
    model = vgg16(pretrained=True)
    # Adjust the classifier for CIFAR-100 (100 classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
    return HuggingFaceStyleWrapper(model)


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
    elif model_name == "customcnn":
        return get_CustomCNN_model()
    elif model_name == "resnet18":
        return get_ResNet18_model()
    elif model_name == "vgg16":
        return get_VGG16_model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
