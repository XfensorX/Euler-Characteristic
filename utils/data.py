from dataclasses import dataclass

import numpy as np
import torch
from skimage import measure
from torch.utils.data import DataLoader, Dataset

from utils.image import create_random_image


class ImageDataset(Dataset):
    def __init__(self, num_samples, img_x_size, img_y_size, img_black_prob):
        self.num_samples = num_samples
        self.samples = np.array(
            [
                create_random_image(
                    x_size=img_x_size, y_size=img_y_size, prob_black=img_black_prob
                )
                for _ in range(num_samples)
            ],
            dtype="float32",
        )
        self.euler_chars = np.array(
            [[measure.euler_number(x)] for x in self.samples], dtype="float32"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.euler_chars[idx]
        return sample, label


@dataclass
class SplittedDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_splitted_dataloader(
    dataset, train_set_perc, val_set_perc, batch_size, device="cpu"
) -> SplittedDataLoaders:
    """
    Creates data loaders and torch.util.data.Datasets
    split by the percentages given for training and validation
    """

    train_size = int(train_set_perc / 100 * len(dataset))
    val_size = int(val_set_perc / 100 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if not test_size > 0:
        raise ValueError(
            "No test set left. Please adjust training/validation/test split fractions."
        )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator(device=device),
    )

    return SplittedDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        ),
        validation=DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator(device=device),
        ),
        test=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator(device=device),
        ),
    )
