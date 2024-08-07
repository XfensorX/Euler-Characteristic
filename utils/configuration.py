import enum
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import yaml

from utils.errors import InvalidOptionError
from utils.config import dtype_mapping
import numpy as np


def set_torch_defaults(device: str, dtype: str, num_threads: int):
    if dtype not in dtype_mapping:
        raise ValueError(f"{dtype} not available. Choose from: {", ".join(dtype_mapping)}")

    torch.manual_seed(42)
    np.random.seed(42)

    torch.set_num_threads(num_threads)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype_mapping[dtype])


class Criterion(enum.Enum):
    MSE_LOSS = "MSELoss"

    def get_func(self):
        if self is Criterion.MSE_LOSS:
            return torch.nn.MSELoss()

        raise NotImplementedError(f"Criterion {self} is not implemented.")

    def __str__(self):
        return str(self.value)

    @staticmethod
    def check_if_valid(criterion: str):
        if not criterion in [c.value for c in Criterion]:
            raise InvalidOptionError(criterion, [c.value for c in Criterion])


class Optimizer(enum.Enum):
    ADAM = "Adam"

    def get_func(self, model_parameters, learning_rate):
        if self is Optimizer.ADAM:
            return torch.optim.Adam(model_parameters, lr=learning_rate)

        raise NotImplementedError(f"Optimizer {self} is not implemented.")

    def __str__(self):
        return str(self.value)

    @staticmethod
    def check_if_valid(optimizer: str):
        if not optimizer in [o.value for o in Optimizer]:
            raise InvalidOptionError(optimizer, [o.value for o in Optimizer])


@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int

    @staticmethod
    def get_dummy():
        return TrainingConfig(0.01, 50)


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

    def to_yaml_file(self, path: str):
        with open(path, "w", encoding="utf8") as config_file:
            yaml.dump(asdict(self), config_file)

    def __str__(self):
        return yaml.dump(asdict(self))

    @staticmethod
    def get_dummy(model_names: List[str]):
        return ExperimentConfig(
            28,
            28,
            0.8,
            10000,
            0.8,
            0.1,
            32,
            "MSELoss",
            "Adam",
            {name: TrainingConfig.get_dummy() for name in model_names},
        )

    @staticmethod
    def save_dummy(path: str, model_names: List[str]):
        dummy_path = os.path.join(path, "dummy_config.yaml")
        with open(dummy_path, "w") as config_file:
            config_file.write(str(ExperimentConfig.get_dummy(model_names)))


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
        raise error from e

    except InvalidOptionError as e:
        error = ValueError("Invalid yaml file.")
        error.note = str(e) + str(e.note)
        raise error from e

    return config
