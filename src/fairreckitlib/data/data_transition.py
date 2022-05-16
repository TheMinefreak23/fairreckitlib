"""This module contains a data transition definition.

Classes:

    DataTransition: data descriptions to be used between pipelines.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Tuple

from dataclasses import dataclass

from .set.dataset import Dataset


@dataclass
class DataTransition:
    """Data Transition to transfer pipeline data."""

    dataset : Dataset
    output_dir: str
    train_set_path: str
    test_set_path: str
    rating_scale: Tuple[float, float]
    rating_type: str
