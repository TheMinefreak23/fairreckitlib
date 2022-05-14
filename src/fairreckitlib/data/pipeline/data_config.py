"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Optional

from ..ratings.convert_config import ConvertConfig
from ..split.split_config import SplitConfig

KEY_DATASETS = 'datasets'
KEY_DATA_FILTERS = 'filters'


@dataclass
class DatasetConfig:
    """Dataset Configuration."""

    name: str
    prefilters: []
    converter: Optional[ConvertConfig]
    splitting: SplitConfig
