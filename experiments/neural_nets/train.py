from itertools import chain

from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig


def get_models(config: ExperimentConfig):
    hidden_layer_configs = list(
        chain.from_iterable(
            [[[n] * x for x in range(1, 10)] for n in range(10, 100, 10)]
        )
    )

    return {
        f"in-{'-'.join([str(x) for x in hidden_layer_config])}-out": VanillaNN(
            config.img_x_size * config.img_y_size,
            1,
            hidden_layer_config,
            flatten_input=True,
        )
        for hidden_layer_config in hidden_layer_configs
    }


class Experiment(experiment.Experiment):
    pass
