from models.VanillaNN import VanillaNN
from utils import experiment
from utils.configuration import ExperimentConfig
from models.VanillaConvNN import VanillaConvNN
from models.VanillaTrans import TransformerBlock
from models.VanillaTrans import Transformer
from models.VanillaVTrans import VisionTransformer


def get_models(config: ExperimentConfig):
    return {
 "vit": VisionTransformer(
    img_size=12,  # width and height of the image
    patch_size=4,  # size of each image patch
    embed_size=128,  # size of each patch embedding
    output_size=2,  # the output size, depends on the specific task
    heads=6,  # number of attention heads
    num_layers=4 # Number of Transformer blocks
        ) 
    }


class Experiment(experiment.Experiment):
    pass