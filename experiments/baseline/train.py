from utils import experiment
from utils.configuration import ExperimentConfig
from models.Constant import Constant


def get_models(config: ExperimentConfig):
    return {"baseline": Constant(12.2845)}


class Experiment(experiment.Experiment):
    pass
