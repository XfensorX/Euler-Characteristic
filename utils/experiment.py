from abc import abstractmethod

from utils.data import ImageDataset, SplittedDataLoaders, create_splitted_dataloader
from utils.configuration import ExperimentConfig
from utils.training import train_model


class Experiment:
    # TODO: How to use several models?
    def __init__(self, config: ExperimentConfig, model):
        self.config = config
        self.model = model
        self.reset_experiment()

    @abstractmethod
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
        self.optimizer = self.config.get_optimizer_func(self.model.parameters())

    def run(self, with_reset=True, output_to=print):
        if with_reset:
            self.reset_experiment()

        train_losses, val_losses = train_model(
            self.model,
            self.data_loaders,
            self.criterion,
            self.optimizer,
            num_epochs=self.config.epochs,
            # TODO: To config? Or maybe better just log it
            verbose=False,
            output_to=output_to,
        )

        return train_losses, val_losses
