import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from utils.experiment import (
    ModelExperimentResult,
    TrainingHistory,
    WholeExperimentResult,
)


def generate_results(results: WholeExperimentResult, path: str):
    os.makedirs(path, exist_ok=True)

    for model, info in results.model_results.items():
        save_individual_model_results(model, info, path)

    results.config.to_yaml_file(os.path.join(path, "used_config.yaml"))
    combined_metrics_to_csv(
        results.model_results, os.path.join(path, "combined_losses.csv")
    )

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


def combined_metrics_to_csv(model_results: Dict[str, ModelExperimentResult], path: str):
    pd.DataFrame.from_dict(
        {
            model: [
                info.losses.train,
                info.losses.validation,
                info.losses.test,
                info.losses_with_rounding.train,
                info.losses_with_rounding.validation,
                info.losses_with_rounding.test,
                info.parameters,
            ]
            for model, info in model_results.items()
        },
        orient="index",
        columns=pd.MultiIndex.from_product(
            [["Loss", "Loss with Rounding"], ["Train", "Validation", "Test"]]
        ).union(pd.MultiIndex.from_tuples([("Model Info", "Parameters")])),
    ).to_csv(path)


def save_individual_model_results(
    model_name: str, model_result: ModelExperimentResult, directory: str
):
    model_path = os.path.join(directory, model_name.replace(" ", "_"))
    os.makedirs(model_path, exist_ok=True)

    model_result.history.to_csv_file(os.path.join(model_path, "training_history.csv"))

    plot_history(
        model_histories={model_name: model_result.history},
        title=f"Training History of '{model_name}'",
        save_path=os.path.join(model_path, "training_history.png"),
    )

    with open(os.path.join(model_path, "description.txt"), "w", encoding="utf8") as f:
        f.write(model_result.description)


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
