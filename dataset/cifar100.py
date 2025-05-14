import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class CustomCIFAR100Dataset(Dataset):
    def __init__(self, dataset_split, resize=(32, 32), normalize=True):
        self.dataset = dataset_split
        self.resize = resize
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load image using PIL
        image = self.dataset[idx]["img"]

        # Resize the image manually
        image = image.resize(self.resize)

        # Convert the image to numpy array (HWC)
        image = np.array(image)

        # Normalize the image (if specified)
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
            std = np.array([0.229, 0.224, 0.225])  # ImageNet std
            image = (
                image / 255.0 - mean
            ) / std

        image = (
            torch.tensor(image).permute(2, 0, 1).float()
        )

        label = self.dataset[idx]["label"]

        return image, label


def get_cifar100_ds(config):
    dataset = load_dataset("cifar100")
    train_dataset = CustomCIFAR100Dataset(dataset["train"])
    val_dataset = CustomCIFAR100Dataset(dataset["test"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], resize=(
            config["ds_width"], config["ds_height"]), shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], resize=(
            config["ds_width"], config["ds_height"]), shuffle=False)
    return train_loader, val_loader
