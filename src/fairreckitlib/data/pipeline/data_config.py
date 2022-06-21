"""This module contains the dataset configuration.

Classes:

    DatasetConfig: dataset configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..filter.filter_config import DataSubsetConfig
from ..ratings.convert_constants import KEY_RATING_CONVERTER
from ..ratings.convert_config import ConvertConfig
from ..split.split_constants import KEY_SPLITTING
from ..split.split_config import SplitConfig


@dataclass
class DataMatrixConfig(DataSubsetConfig):
    """Data Matrix Configuration.

    dataset: the name of the dataset.
    matrix: the name of the dataset matrix.
    filter_passes: the subset of the dataset matrix as a list of filter passes.
    converter: the rating converter of the dataset matrix.
    splitting: the train/test splitter of the dataset matrix.
    """

    converter: Optional[ConvertConfig]
    splitting: SplitConfig

    def get_data_matrix_name(self) -> str:
        """Get the combined dataset and matrix name of the configuration."""
        return self.dataset + '_' + self.matrix

    def to_yml_format(self) -> Dict[str, Any]:
        """Format data matrix configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset configuration.
        """
        yml_format = DataSubsetConfig.to_yml_format(self)
        yml_format[KEY_SPLITTING] = self.splitting.to_yml_format()
        # only include rating modifier if it is present
        if self.converter:
            yml_format[KEY_RATING_CONVERTER] = self.converter.to_yml_format()

        return yml_format
