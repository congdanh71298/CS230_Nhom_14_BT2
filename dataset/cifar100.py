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

        label = self.dataset[idx]['fine_label']

        return image, label


def get_cifar100_ds(config):
    dataset = load_dataset("cifar100")

    train_dataset = CustomCIFAR100Dataset(dataset["train"], resize=(
        config["ds_width"], config["ds_height"]))
    test_dataset = CustomCIFAR100Dataset(dataset["test"], resize=(
        config["ds_width"], config["ds_height"]))

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],  shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, test_loader
