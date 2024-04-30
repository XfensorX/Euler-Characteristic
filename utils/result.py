import os
from dataclasses import asdict
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from utils.experiment import TrainingHistory, WholeExperimentResult


# TODO: refactor this into individual subfunctions
def generate_results(results: WholeExperimentResult, path: str):
    os.makedirs(path, exist_ok=True)

    config_yaml_path = os.path.join(path, "used_config.yaml")
    with open(config_yaml_path, "w", encoding="utf8") as config_file:
        yaml.dump(asdict(results.config), config_file)

    for model_name, model_result in results.model_results.items():
        model_folder_path = os.path.join(path, model_name.replace(" ", "_"))
        os.makedirs(model_folder_path, exist_ok=True)

        loss_history_path = os.path.join(model_folder_path, f"loss_history.csv")
        pd.DataFrame(
            {
                "Epoch": list(range(1, 1 + len(model_result.history.train))),
                "Training Loss": model_result.history.train,
                "Validation Loss": model_result.history.validation,
            }
        ).to_csv(loss_history_path, index=False)

        plot_history(
            model_histories={model_name: model_result.history},
            title=f"Training History of '{model_name}'",
            save_path=os.path.join(model_folder_path, "training_history.png"),
        )
        with open(
            os.path.join(model_folder_path, "description.txt"), "w", encoding="utf8"
        ) as f:
            f.write(model_result.description)

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

    plot_history(
        {model: result.history for model, result in results.model_results.items()},
        title="Train Losses",
        save_path=os.path.join(path, "train_loss.png"),
        plot_validation_loss=False,
    )
    plot_history(
        {model: result.history for model, result in results.model_results.items()},
        title="Validation Losses",
        save_path=os.path.join(path, "val_loss.png"),
        plot_training_loss=False,
    )


def plot_history(
    model_histories: Dict[str, TrainingHistory],
    title="",
    save_path=None,
    plot_training_loss=True,
    plot_validation_loss=True,
):
    if plot_training_loss:
        for model_name, history in model_histories.items():
            plt.plot(history.train, label=f"Train ({model_name})")

    if plot_validation_loss:
        for model_name, history in model_histories.items():
            plt.plot(history.validation, label=f"Validation ({model_name})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=1000)
        plt.close()
    else:
        plt.show()
