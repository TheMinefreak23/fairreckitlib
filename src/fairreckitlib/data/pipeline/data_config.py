"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any

from ..ratings.base_converter import DataModifier

KEY_DATASETS = 'datasets'
KEY_DATA_FILTERS = 'filters'


@dataclass
class SplitConfig:
    """Dataset Splitting Configuration."""

    test_ratio: float
    type: str
    params: {str: Any}


@dataclass
class DatasetConfig:
    """Dataset Configuration."""

    name: str
    prefilters: []
    rating_modifier: DataModifier
    splitting: SplitConfig
