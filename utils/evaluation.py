from dataclasses import dataclass

import torch

from utils.data import SplittedDataLoaders


@dataclass
class LossCalculation:
    train: float
    validation: float
    test: float


def calculate_all_losses(
    data_loaders: SplittedDataLoaders, model, criterion, use_rounding=False
) -> LossCalculation:

    return LossCalculation(
        train=calculate_loss(
            model, data_loaders.train, criterion, use_rounding=use_rounding
        ),
        validation=calculate_loss(
            model, data_loaders.validation, criterion, use_rounding=use_rounding
        ),
        test=calculate_loss(
            model, data_loaders.test, criterion, use_rounding=use_rounding
        ),
    )


def calculate_loss(model, data_loader, criterion, use_rounding=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            if use_rounding:
                outputs = torch.round(outputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features.size(0)

    return total_loss / len(data_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
