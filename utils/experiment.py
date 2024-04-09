from abc import ABCMeta, abstractmethod

from torch import nn

from utils.configuration import ExperimentConfig
from utils.data import ImageDataset, SplittedDataLoaders, create_splitted_dataloader
from utils.training import train_model


class Experiment(metaclass=ABCMeta):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.check_model_configurations_are_present()
        self.reset_experiment()

    @property
    @abstractmethod
    def models(self) -> {str: nn.Module}:
        pass

    def post_dataset_creation(self):
        pass

    def reset_experiment(self):
        self.dataset = ImageDataset(
            num_samples=self.config.dataset_size,
            img_x_size=self.config.img_x_size,
            img_y_size=self.config.img_y_size,
            img_black_prob=self.config.img_prob_black,
        )

        self.post_dataset_creation()

        self.data_loaders: SplittedDataLoaders = create_splitted_dataloader(
            self.dataset,
            self.config.train_set_perc,
            self.config.validation_set_perc,
            self.config.batch_size,
        )
        self.criterion = self.config.get_criterion_func()
        self.optimizers = {
            name: self.config.get_optimizer_func(name, model.parameters())
            for name, model in self.models.items()
        }

    def check_model_configurations_are_present(self):
        if not all(
            [key in self.config.training_configs.keys() for key in self.models.keys()]
        ):
            raise ValueError("Not all registered models have a configuration present.")

    def run(self, with_reset=True, output_to=print, model=None):
        if with_reset:
            self.reset_experiment()

        models_to_run = self.models.keys()
        # TODO: Add custom error class
        if model is not None:
            if not model in self.models.keys():
                error = KeyError()
                error.note = "Invalid Model Name, choose from: \n- " + "\n- ".join(
                    self.models.keys()
                )
                raise error
            models_to_run = [model]

        losses = {}
        for name in models_to_run:
            output_to(f"Running {name}...")
            train_losses, val_losses = train_model(
                self.models[name],
                self.data_loaders,
                self.criterion,
                self.optimizers[name],
                num_epochs=self.config.training_configs[name].epochs,
                # TODO: To config? Or maybe better just log it
                verbose=False,
                output_to=output_to,
            )
            losses[name] = (train_losses, val_losses)
            output_to("")

        return losses
