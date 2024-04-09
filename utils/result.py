import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from utils.evaluation import calculate_loss
from utils.visuals import save_image_as_img


def save_predictions(
    data_loader,
    model,
    use_rounding=False,
    save_all_images=False,
    save_wrong_images=False,
):
    folder_name = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(folder_name, exist_ok=True)

    df = pd.DataFrame(columns=["index", "euler_number", "model_output", "difference"])
    os.makedirs(os.path.join(folder_name, "images"), exist_ok=True)

    index = 0
    for features, labels in data_loader:
        outputs = model(features)
        if use_rounding:
            outputs = torch.round(outputs)

        for feature, label, output in zip(features, labels, outputs):
            image_name = f"image{index}.png"

            if save_all_images or (save_wrong_images and label != output):
                image_path = os.path.join(folder_name, "images", image_name)
                save_image_as_img(feature, file_name=image_path, title=image_name)

            df.loc[index] = [
                image_name,
                label.item(),
                output.item(),
                (output - label).item(),
            ]
            index += 1

    df.to_csv(os.path.join(folder_name, "results.csv"), index=False)


def plot_training_history(train_losses, val_losses, title=""):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_losses(data_loaders, criterion, models):
    """
    models is a dictionary mapping model name to the model
    """
    names, losses_test, losses_val, losses_train = [], [], [], []
    for name, model in models.items():
        names.append(name)
        losses_test.append(calculate_loss(model, data_loaders["test"], criterion))
        losses_train.append(calculate_loss(model, data_loaders["train"], criterion))
        losses_val.append(calculate_loss(model, data_loaders["validation"], criterion))

    plt.bar(names, losses_test, label="Test")
    plt.bar(names, losses_val, label="Validation")
    plt.bar(names, losses_train, label="Test")
    plt.xlabel("Model")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.xticks(rotation=45)
    plt.show()
