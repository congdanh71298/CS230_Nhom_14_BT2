import os
from pathlib import Path
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset.cifar100 import get_cifar100_ds
from utils.log_to_file import log_to_file
from utils.weight_retrieve import get_weights_file_path, latest_weights_file_path
from sklearn.metrics import precision_score, recall_score, f1_score


def cal_test_metrics(model, test_loader, device, print_msg=print):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # shape: (batch_size, seq_len, d_model)
            logits = outputs.logits
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print_msg(f"Accuracy:  {accuracy:.2%}")
    print_msg(f"Precision: {precision:.2%}")
    print_msg(f"Recall:    {recall:.2%}")
    print_msg(f"F1-Score:  {f1:.2%}")

    return accuracy, precision, recall, f1, all_preds, all_labels


def finetuning(config, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    (train_data_loader, val_data_loader) = get_cifar100_ds(config)

    model.to(device)

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

    os.makedirs("log", exist_ok=True)

    log_file = os.path.join("log", f"{config['model_folder']}.txt")

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_data_loader, desc=f"Processing epoch {epoch:02d}")
        epoch_loss = 0.0
        epoch_start_time = time.time()
        for batch in batch_iterator:
            model.train()
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            logits = outputs.logits

            loss = loss_fn(logits, labels)
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        epoch_end_time = time.time()

        accuracy, precision, recall, f1, _, _ = cal_test_metrics(
            model,
            val_data_loader,
            device,
            lambda msg: batch_iterator.write(msg),
        )
        epoch_duration = epoch_end_time - epoch_start_time  # Time in seconds

        avg_epoch_loss = epoch_loss / len(train_data_loader)
        log_message = f"Epoch: {epoch+1}; Avg Loss: {avg_epoch_loss:.4f}; Test acc: {accuracy:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1: {f1:.4f}; Duration: {epoch_duration:.2f}s; Device: {device}"
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
