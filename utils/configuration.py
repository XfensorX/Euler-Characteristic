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

        raise ValueError("An Option is not implemented.")

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

        raise ValueError("An Option is not implemented.")

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
        return self.criterion.get_func()

    def get_optimizer_func(self, model_parameters):
        return self.optimizer.get_func(model_parameters, self.learning_rate)

    def __str__(self):
        config_dict = {field: getattr(self, field) for field in self.__annotations__}
        yaml_str = "\n".join(
            [f"- {key}: {value}" for key, value in config_dict.items()]
        )
        return yaml_str


def load_config(path: str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    try:
        config = ExperimentConfig(**config)
    except TypeError:
        error = ValueError("Invalid yaml file.")
        error.note = (
            f"Wrong Format in the configuration of {path}. "
            + "The following configurations are missing:\n"
            + ":\n".join(str(e).split("'")[1::2] + [""])
        )
        raise error

    except ValueError as e:
        error = ValueError("Invalid yaml file.")
        error.note = str(e) + str(e.note)
        raise error

    return config
