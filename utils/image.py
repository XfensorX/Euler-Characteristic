import numpy as np


def create_images(x_size, y_size, prob_black, num_images, seed=42):
    np.random.seed(seed)
    return [create_random_image(x_size, y_size, prob_black) for _ in range(num_images)]


def create_random_image(x_size, y_size, prob_black):
    return np.random.choice(
        [0.0, 1.0], size=(y_size, x_size), p=[prob_black, 1 - prob_black]
    )
