from utils import experiment
from utils.configuration import ExperimentConfig
from models.SkipConnectionConv import SkipConnectionConv


def get_models(config: ExperimentConfig):
    return {
        "skip_conv": SkipConnectionConv(
            img_x_size=config.img_x_size,
            img_y_size=config.img_y_size,
        ),
    }


class Experiment(experiment.Experiment):
    pass
