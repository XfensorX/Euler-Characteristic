from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig


def get_models(config: ExperimentConfig):
    return {
        "in-20-out": VanillaNN(
            config.img_x_size * config.img_y_size,
            1,
            [20],
            flatten_input=True,
        ),
        "in-20-20-20-out": VanillaNN(
            config.img_x_size * config.img_y_size,
            1,
            [20, 20, 20],
            flatten_input=True,
        ),
    }


class Experiment(experiment.Experiment):
    pass
