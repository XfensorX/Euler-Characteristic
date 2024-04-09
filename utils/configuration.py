import enum
import os
from dataclasses import dataclass

import torch
import yaml


class Criterion(enum.Enum):
    MSE_LOSS = "MSELoss"

    def __str__(self):
        return self.value

    def get_func(self):
        if self is Criterion.MSE_LOSS:
            return torch.nn.MSELoss()

    @staticmethod
    def check_if_valid(criterion: str):
        if criterion in [c.value for c in Criterion]:
            return

        error = ValueError("Invalid criterion in configuration.")
        error.note = "Valid criteria are: \n- " + "\n- ".join(
            [c.value for c in Criterion]
        )
        raise error


class Optimizer(enum.Enum):
    ADAM = "Adam"

    def __str__(self):
        return self.value

    def get_func(self, model_parameters, learning_rate):
        if self is Optimizer.ADAM:
            return torch.optim.Adam(model_parameters, lr=learning_rate)

    @staticmethod
    def check_if_valid(optimizer: str):
        if optimizer in [o.value for o in Optimizer]:
            return

        error = ValueError("Invalid optimizer in configuration.")
        error.note = "Valid optimizer are: \n- " + "\n- ".join(
            [o.value for o in Optimizer]
        )
        raise error


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

    def __post_init__(self):
        Optimizer.check_if_valid(self.optimizer)
        Criterion.check_if_valid(self.criterion)
        self.optimizer = Optimizer(self.optimizer)
        self.criterion = Criterion(self.criterion)

    def get_criterion_func(self):
        self.criterion.get_func()

    def get_optimizer_func(self, model_parameters):
        self.optimizer.get_func(model_parameters, self.learning_rate)

    def __str__(self):
        config_dict = {field: getattr(self, field) for field in self.__annotations__}
        yaml_str = "\n".join(
            [f"- {key}: {value}" for key, value in config_dict.items()]
        )
        return yaml_str


def load_config(experiment_dir: str, config_file: str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.
    """
    config_path = os.path.join(experiment_dir, config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return ExperimentConfig(**config)
