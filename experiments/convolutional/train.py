from utils import experiment
from utils.configuration import ExperimentConfig
from models.GoodConvolutional import GoodConvolutional


def get_models(config: ExperimentConfig):
    return {
        "conv": GoodConvolutional(
            img_x_size=config.img_x_size,
            img_y_size=config.img_y_size,
        ),
    }


class Experiment(experiment.Experiment):
    pass
