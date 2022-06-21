"""This module contains the subgroup/filter configurations.

Classes:

    FilterConfig: data filter configuration.
    FilterPassConfig: data filter pass configuration consisting of multiple filters.
    DataSubsetConfig: data subset configuration consisting of multiple filter passes.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ...core.config.config_object import ObjectConfig
from ...core.config.config_yml import YmlConfig, format_yml_config_list
from ..set.dataset_constants import KEY_DATASET, KEY_MATRIX
from .filter_constants import KEY_DATA_FILTER_PASS, KEY_DATA_SUBSET


@dataclass
class FilterConfig(ObjectConfig):
    """Filter Configuration.

    name: the name of the filter.
    params: the parameters of the filter.
    """


@dataclass
class FilterPassConfig(YmlConfig):
    """Filter Pass Configuration.

    The pass consists of multiple filters that are applied in order.

    filters: list of filter configurations.
    """

    filters: List[FilterConfig]

    def to_yml_format(self) -> Dict[str, Any]:
        """Format filter pass configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the subgroup configuration.
        """
        return {KEY_DATA_FILTER_PASS: format_yml_config_list(self.filters)}


@dataclass
class DataSubsetConfig:
    """Data Subset Configuration.

    The subset of the data consists of multiple filter passes that are applied
    on the dataset individually, and thereafter they are merged to create the subset.

    dataset: the name of the dataset.
    matrix: the name of the dataset matrix.
    filter_passes: the subset as a list of filter passes.
    """

    dataset: str
    matrix: str
    filter_passes: List[FilterPassConfig]

    def to_yml_format(self) -> Dict[str, Any]:
        """Format data subset configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset configuration.
        """
        yml_format = {KEY_DATASET: self.dataset, KEY_MATRIX: self.matrix}
        # only include filter passes if it has entries
        if len(self.filter_passes) > 0:
            yml_format[KEY_DATA_SUBSET] = format_yml_config_list(self.filter_passes)

        return yml_format
