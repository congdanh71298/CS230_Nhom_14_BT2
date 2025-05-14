import os
from pathlib import Path
import time
from transformers import ViTForImageClassification, ViTConfig
from ViT.config import get_config
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.cifar100 import get_cifar100_ds
from utils import log_to_file
from utils.weight_retrieve import get_weights_file_path, latest_weights_file_path


def get_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=100)
    return model


def run_validation(model, validation_loader, device, print_msg=print):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # shape: (batch_size, seq_len, d_model)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print_msg(f"Accuracy: {accuracy:.2%}")
    return accuracy


def finetuning():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    (train_data_loader, val_data_loader) = get_cifar100_ds(config)

    model = get_model()
    model.to(device)
    print(f"Model total params {model.num_params}")

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss().to(device)

    log_file = os.path.join(config["model_folder"], "training_log.txt")
    # Header for the lo

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(
            train_data_loader, desc=f"Processing epoch {epoch:02d}")
        epoch_loss = 0.0

        epoch_start_time = time.time()
        for batch in batch_iterator:
            model.train()
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        epoch_end_time = time.time()

        val_accuracy = run_validation(
            model,
            val_data_loader,
            device,
            lambda msg: batch_iterator.write(msg),
        )
        epoch_duration = epoch_end_time - epoch_start_time  # Time in seconds

        avg_epoch_loss = epoch_loss / len(train_data_loader)
        log_message = f"Epoch: {epoch+1}; Avg Loss: {avg_epoch_loss:.4f}; Val acc: {val_accuracy:.4f}; Duration: {epoch_duration:.2f}s; Device: {device}"
        log_to_file(log_file, log_message)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
