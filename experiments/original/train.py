from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig
from models.VanillaConvNN import VanillaConvNN


def get_models(config: ExperimentConfig):
    return {
        "in-20-out": VanillaNN(
            config.img_x_size * config.img_y_size,
            1,
            [20],
            flatten_input=True,
        ),
        "in-20-20-20-out": VanillaNN(
            input_size=config.img_x_size * config.img_y_size,
            output_size=1,
            hidden_sizes=[20, 20, 20],
            flatten_input=True,
        ),
        "conv": VanillaConvNN(
            hidden_size=28,
            output_size=1,
            img_x_size=config.img_x_size,
            img_y_size=config.img_y_size,
        ),
    }


class Experiment(experiment.Experiment):
    pass
