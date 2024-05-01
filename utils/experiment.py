from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from torch import nn

from utils.configuration import ExperimentConfig
from utils.data import ImageDataset, SplittedDataLoaders, create_splitted_dataloader
from utils.evaluation import LossCalculation, calculate_all_losses, count_parameters
from utils.training import train_model


@dataclass
class TrainingHistory:
    train: List[float]
    validation: List[float]

    def to_csv_file(self, path: str):
        pd.DataFrame(
            {
                "Epoch": list(range(1, 1 + len(self.train))),
                "Training Loss": self.train,
                "Validation Loss": self.validation,
            }
        ).to_csv(path, index=False)


@dataclass
class ModelExperimentResult:
    history: TrainingHistory
    losses: LossCalculation
    losses_with_rounding: LossCalculation
    parameters: int
    description: str


@dataclass
class WholeExperimentResult:
    model_results: Dict[str, ModelExperimentResult]
    config: ExperimentConfig


class Experiment:
    def __init__(
        self,
        config: ExperimentConfig,
        models: {str: nn.Module},
        device="cpu",
        dtype="float32",
    ):
        self.device = device
        self.config = config
        self.models = models
        self.dtype = dtype
        self.check_model_configurations_are_present()
        self.reset_experiment()

    def post_dataset_creation(self):
        pass

    def reset_experiment(self):
        self.dataset = ImageDataset(
            num_samples=self.config.dataset_size,
            img_x_size=self.config.img_x_size,
            img_y_size=self.config.img_y_size,
            img_black_prob=self.config.img_prob_black,
            dtype=self.dtype,
        )

        self.post_dataset_creation()

        self.data_loaders: SplittedDataLoaders = create_splitted_dataloader(
            self.dataset,
            self.config.train_set_perc,
            self.config.validation_set_perc,
            self.config.batch_size,
            device=self.device,
        )
        self.criterion = self.config.get_criterion_func()
        self.optimizers = {
            name: self.config.get_optimizer_func(name, model.parameters())
            for name, model in self.models.items()
        }

    def check_model_configurations_are_present(self):
        if all(
            [key in self.config.training_configs.keys() for key in self.models.keys()]
        ):
            return

        if "rest" in self.config.training_configs.keys():
            for key in self.models.keys():
                if key in self.config.training_configs:
                    return
                else:
                    self.config.training_configs[key] = self.config.training_configs[
                        "rest"
                    ]
        else:

            raise ValueError("Not all registered models have a configuration present.")

    def run(
        self, with_reset=True, output_to=print, model=None
    ) -> WholeExperimentResult:
        if with_reset:
            self.reset_experiment()

        models_to_run = self.models.keys()
        # TODO: Add custom error class
        if model is not None:
            if not model in self.models.keys():
                error = KeyError()
                error.note = "Invalid Model Name, choose from: \n- " + "\n- ".join(
                    self.models.keys()
                )
                raise error
            models_to_run = [model]

        model_results = {}
        #TODO: Change to enumerate
        for name in models_to_run:
            output_to(f"Running {name}...")
            model = self.models[name]
            train_losses, val_losses = train_model(
                model,
                self.data_loaders,
                self.criterion,
                self.optimizers[name],
                num_epochs=self.config.training_configs[name].epochs,
                output_to=output_to,
            )
            model_results[name] = ModelExperimentResult(
                history=TrainingHistory(train=train_losses, validation=val_losses),
                losses=calculate_all_losses(
                    self.data_loaders,
                    model,
                    self.criterion,
                    use_rounding=False,
                ),
                losses_with_rounding=calculate_all_losses(
                    self.data_loaders,
                    model,
                    self.criterion,
                    use_rounding=True,
                ),
                parameters=count_parameters(model),
                description=str(model),
            )
            output_to("")

        return WholeExperimentResult(config=self.config, model_results=model_results)
