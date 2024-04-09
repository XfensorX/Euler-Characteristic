import enum
import os
from dataclasses import dataclass

import torch
import yaml


class Criterion(enum.Enum):
    MSE_LOSS = "MSELoss"


class Optimizer(enum.Enum):
    ADAM = "Adam"


@dataclass
class ExperimentConfig:
    img_x_size: int
    img_y_size: int
    img_prob_black: float
    dataset_size: int
    train_set_perc: float
    validation_set_perc: float
    batch_size: int
    learning_rate: float
    epochs: int
    criterion: Criterion
    optimizer: Optimizer

    def get_criterion_func(self):
        self.check_for_correct_criterion()

        if self.criterion == Criterion.MSE_LOSS.value:
            return torch.nn.MSELoss()

    def get_optimizer_func(self, model_parameters):
        self.check_for_correct_optimizer()

        if self.optimizer == Optimizer.ADAM.value:
            return torch.optim.Adam(model_parameters, lr=self.learning_rate)

    def check_for_correct_optimizer(self):
        if self.optimizer in [e.value for e in Optimizer]:
            return
        error = ValueError("Invalid optimizer in configuration.")
        error.note = "Valid optimizer are: \n- " + "\n- ".join(
            [o.value for o in Optimizer]
        )
        raise error

    def check_for_correct_criterion(self):
        if self.criterion in [e.value for e in Criterion]:
            return

        error = ValueError("Invalid criterion in configuration.")
        error.note = "Valid criteria are: \n- " + "\n- ".join(
            [c.value for c in Criterion]
        )
        raise error

    def __post_init__(self):
        self.check_for_correct_criterion()
        self.check_for_correct_optimizer()

    def __str__(self):
        config_dict = {field: getattr(self, field) for field in self.__annotations__}
        yaml_str = "\n".join(
            [f"- {key}: {value}" for key, value in config_dict.items()]
        )
        return yaml_str


def load_config(experiment_dir: str, config_file: str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    :param experiment_dir: Directory containing experiment configuration
    :param config_file: Experiment configuration file
    :return: Experiment configuration
    """
    config_path = os.path.join(experiment_dir, config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return ExperimentConfig(**config)
