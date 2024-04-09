from utils import experiment

from models.VanillaNN import VanillaNN


class Experiment(experiment.Experiment):
    # TODO: find some smoother way to accomplish that models are not getting re=-generated at every call
    _models = None

    @property
    def models(self):
        if self._models is None:
            self._models = {
                "Neural Net": VanillaNN(
                    self.config.img_x_size * self.config.img_y_size,
                    1,
                    [20],
                    flatten_input=True,
                ),
                "Bigger Neural Net": VanillaNN(
                    self.config.img_x_size * self.config.img_y_size,
                    1,
                    [20, 20, 20],
                    flatten_input=True,
                ),
            }
        return self._models
