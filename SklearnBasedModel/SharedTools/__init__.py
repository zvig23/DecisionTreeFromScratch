import numpy as np
from sklearn.model_selection import train_test_split


def bootstrap_sample(X, Y, sample_size=0.8, random_seed=None):
    """
    Function that creates a random sample from the given dataset.

    Parameters:
    - X: Features
    - Y: Labels
    - sample_size: Size of the sample to be created
    - random_seed: Seed for random number generation (optional)

    Returns:
    - X_sample: Random sample of features
    - Y_sample: Corresponding random sample of labels
    """
    subsample_X, _, subsample_Y, _ = train_test_split(X, Y, train_size=sample_size, random_state=random_seed)

    return subsample_X, subsample_Y
