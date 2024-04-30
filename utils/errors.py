from utils.configuration import Criterion, Optimizer


class InvalidCriterionError(Exception):
    def __init__(self, criterion):
        self.criterion = criterion
        self.note = "Valid criteria are: \n- " + "\n- ".join(
            [c.value for c in Criterion]
        )

    def __str__(self):
        return (
            f"Invalid criterion '{self.criterion}' in configuration."
            + "\n\n Note: \n"
            + self.note
        )


class InvalidOptimizerError(Exception):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.note = "Valid optimizer are: \n- " + "\n- ".join(
            [o.value for o in Optimizer]
        )

    def __str__(self):
        return (
            f"Invalid optimizer '{self.optimizer}' in configuration."
            + "\n\n Note: \n"
            + self.note
        )
