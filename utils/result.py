import os
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from utils.experiment import WholeExperimentResult
from utils.visuals import save_image_as_img


# TODO: refactor this into individual subfunctions
def generate_results(experiment_name: str, results: WholeExperimentResult, path: str):
    os.makedirs(path, exist_ok=True)

    config_yaml_path = os.path.join(path, "used_config.yaml")
    with open(config_yaml_path, "w") as config_file:
        yaml.dump(asdict(results.config), config_file)

    all_model_data = []
    for model_name, result_obj in results.model_results.items():
        model_folder_path = os.path.join(path, model_name.replace(" ", "_"))
        os.makedirs(model_folder_path, exist_ok=True)

        loss_history_path = os.path.join(model_folder_path, f"loss_history.csv")
        pd.DataFrame(
            {
                "Epoch": list(range(1, 1 + len(result_obj.train_losses_history))),
                "Training Loss": result_obj.train_losses_history,
                "Validation Loss": result_obj.val_losses_history,
            }
        ).to_csv(loss_history_path, index=False)

        plot_training_history(
            result_obj.train_losses_history,
            result_obj.val_losses_history,
            f"Training History of '{model_name}'",
            os.path.join(model_folder_path, "training_history.png"),
        )
        with open(
            os.path.join(model_folder_path, "description.txt"), "w", encoding="utf8"
        ) as f:
            f.write(result_obj.description)

    df_data = {}
    for model, info in results.model_results.items():
        df_data[model] = [
            info.losses.train,
            info.losses.validation,
            info.losses.test,
            info.losses_with_rounding.train,
            info.losses_with_rounding.validation,
            info.losses_with_rounding.test,
            info.parameters,
        ]

    all_losses_df = pd.DataFrame.from_dict(
        df_data,
        orient="index",
        columns=pd.MultiIndex.from_product(
            [["Loss", "Loss with Rounding"], ["Train", "Validation", "Test"]]
        ).union(pd.MultiIndex.from_tuples([("Model Info", "Parameters")])),
    )
    all_losses_df.to_csv(os.path.join(path, "combined_losses.csv"))
    plot_combined_training_history(
        {
            model: result_obj.train_losses_history
            for model, result_obj in results.model_results.items()
        },
        "Train Losses",
        save_path=os.path.join(path, "train_loss.png"),
    )
    plot_combined_training_history(
        {
            model: result_obj.val_losses_history
            for model, result_obj in results.model_results.items()
        },
        "Validation Losses",
        save_path=os.path.join(path, "val_loss.png"),
    )


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


def plot_combined_training_history(losses, val_losses, title="", save_path=None):
    for name, loss in losses.items():
        plt.plot(loss, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=1000)
        plt.close()
    else:
        plt.show()


def plot_training_history(train_losses, val_losses, title="", save_path=None):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=1000)
        plt.close()
    else:
        plt.show()
