from dataclasses import dataclass

import numpy as np
import torch
from skimage import measure
from torch.utils.data import DataLoader, Dataset

from typing import Union

from utils.image import create_random_image


class ImageDataset(Dataset):
    def __init__(self, num_samples, img_x_size, img_y_size, img_black_prob, dtype):
        self.num_samples = num_samples
        self.samples = torch.tensor(
            np.array(
                [
                    create_random_image(
                        x_size=img_x_size, y_size=img_y_size, prob_black=img_black_prob
                    )
                    for _ in range(num_samples)
                ]
            ).astype("float"),
            dtype=dtype,
        )
        self.euler_chars = torch.tensor(
            np.array(
                [[measure.euler_number(x)] for x in np.array(self.samples.cpu())]
            ).astype("float"),
            dtype=dtype,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.euler_chars[idx]
        return sample, label


@dataclass
class SplittedDataLoaders:
    train: Union[DataLoader, torch.Tensor]
    validation: Union[DataLoader, torch.Tensor]
    test: Union[DataLoader, torch.Tensor]


def create_splitted_dataloader(
    dataset,
    train_set_perc,
    val_set_perc,
    batch_size,
    device="cpu",
    use_torch_data_loaders=True,
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

    # TODO: use fact, that sizes can be given as fractions
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator(device=device),
    )

    if use_torch_data_loaders:
        return SplittedDataLoaders(
            train=DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
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
    else:

        train_dataset = dataset[train_dataset.indices]
        val_dataset = dataset[val_dataset.indices]
        test_dataset = dataset[test_dataset.indices]

        train_dataset = split_up_to_batch(
            train_dataset[0], train_dataset[1], batch_size
        )
        test_dataset = split_up_to_batch(test_dataset[0], test_dataset[1], batch_size)
        val_dataset = split_up_to_batch(val_dataset[0], val_dataset[1], batch_size)

        # TODO: Shuffle the training set
        return SplittedDataLoaders(
            train=train_dataset,
            validation=val_dataset,
            test=test_dataset,
        )


def split_up_to_batch(X, Y, batch_size):
    return [
        (
            X[i * batch_size : (i + 1) * batch_size],
            Y[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(len(X) // batch_size + 1)
    ]
