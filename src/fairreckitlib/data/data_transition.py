"""This module contains a data transition definition.

Classes:

    DataTransition: data descriptions to be used between pipelines.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Tuple

from dataclasses import dataclass

from .ratings.convert_constants import RATING_TYPE_THRESHOLD
from .set.dataset import Dataset
from .set.dataset_config import DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT


@dataclass
class DataTransition:
    """Data Transition to transfer pipeline data.

    dataset: the dataset that was used for the data transition.
    matrix_name: the dataset matrix name that was used for the data transition.
    output_dir: the output directory of the data transition.
    train_set_path: the train set path in the output directory.
    test_set_path: the test set path in the output directory.
    rating_scale: the minimum and maximum rating in the train and test set combined.
    """

    dataset : Dataset
    matrix_name: str
    output_dir: str
    train_set_path: str
    test_set_path: str
    rating_scale: Tuple[float, float]

    def get_rating_type(self) -> str:
        """Get the rating type of the data transition.

        Returns:
            the rating type (either 'explicit' or 'implicit').
        """
        return DATASET_RATINGS_IMPLICIT \
            if self.rating_scale[1] > RATING_TYPE_THRESHOLD else DATASET_RATINGS_EXPLICIT
