from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig
from torch import nn


def get_models(config: ExperimentConfig):
    activations = [nn.ReLU, nn.GELU, nn.Tanh]

    return {
        str(activation).split(".")[-1][:-2]: VanillaNN(
            config.img_x_size * config.img_y_size,
            1,
            [40],
            flatten_input=True,
            activation=activation,
        )
        for activation in activations
    }


class Experiment(experiment.Experiment):
    pass
