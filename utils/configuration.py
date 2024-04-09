import enum
from dataclasses import dataclass, asdict

import torch
import yaml
from typing import Dict


class Criterion(enum.Enum):
    MSE_LOSS = "MSELoss"

    def __str__(self):
        return str(self.value)

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
        return str(self.value)

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
class TrainingConfig:
    learning_rate: float
    epochs: int


@dataclass
class ExperimentConfig:
    img_x_size: int
    img_y_size: int
    img_prob_black: float
    dataset_size: int
    train_set_perc: float
    validation_set_perc: float
    batch_size: int
    criterion: str  # Criterion
    optimizer: str  # Optimizer
    training_configs: Dict[str, TrainingConfig]

    def __post_init__(self):
        Optimizer.check_if_valid(self.optimizer)
        Criterion.check_if_valid(self.criterion)

    def get_criterion_func(self):
        return Criterion(self.criterion).get_func()

    def get_optimizer_func(self, model_name, model_parameters):
        return Optimizer(self.optimizer).get_func(
            model_parameters, self.training_configs[model_name].learning_rate
        )

    def __str__(self):
        return yaml.dump(asdict(self))


def load_config(path: str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # TODO: error handling, could be refactored with the other
    if "training_configs" in config:
        config["training_configs"] = {
            name: TrainingConfig(**info)
            for name, info in config["training_configs"].items()
        }

    try:
        config = ExperimentConfig(**config)
    except TypeError as e:
        error = ValueError("Invalid yaml file.")

        if "got an unexpected keyword argument" in str(e):
            error.note = (
                f"Wrong Format in the configuration of {path} \n"
                + "The following configurations are not allowed:\n"
                + ":\n".join(str(e).split("'")[1::2] + [""])
            )
        else:
            error.note = (
                f"Wrong Format in the configuration of {path} \n"
                + "The following configurations are missing:\n"
                + ":\n".join(str(e).split("'")[1::2] + [""])
            )
        raise error

    except ValueError as e:
        error = ValueError("Invalid yaml file.")
        error.note = str(e) + str(e.note)
        raise error

    return config
