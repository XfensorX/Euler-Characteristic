from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig
from models.VanillaConvNN import VanillaConvNN
from models.VanillaTrans import TransformerBlock
from models.VanillaTrans import Transformer
from models.VanillaVTrans import VisionTransformer


def get_models(config: ExperimentConfig):
    return {
        "transformer": Transformer(
            embed_size=144,
            output_size=1,
            img_x_size=config.img_x_size,
            img_y_size=config.img_y_size,
            heads=4,  # You might also want to make this configurable
            num_layers=16  # Set this as per your model complexity requirement
        )
    }


class Experiment(experiment.Experiment):
    pass