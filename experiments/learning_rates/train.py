from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig
from models.VanillaConvNN import VanillaConvNN


def get_models(config: ExperimentConfig):
    learning_rates = [0.1, 0.3, 0.7, 1, 2, 5, 7, 10]
    return {
        str(lr): VanillaNN(
            config.img_x_size * config.img_y_size,
            1,
            [40],
            flatten_input=True,
        )
        for lr in learning_rates
    }


class Experiment(experiment.Experiment):
    pass
