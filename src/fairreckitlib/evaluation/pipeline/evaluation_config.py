"""This module contains the metric configuration.

Classes:

    MetricConfig: metric configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ...core.config.config_object import ObjectConfig
from ...data.filter.filter_constants import KEY_DATA_FILTERS


@dataclass
class MetricConfig(ObjectConfig):
    """Metric Configuration.

    name: the name of the metric.
    params: the parameters of the metric.
    prefilters: the prefilters of the metric.
    """

    prefilters: List[Any]

    def to_yml_format(self) -> Dict[str, Any]:
        """Format metric configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the metric configuration.
        """
        yml_format = ObjectConfig.to_yml_format(self)

        # only include prefilters if it has entries
        if len(self.prefilters) > 0:
            # TODO convert filters to yml format
            yml_format[KEY_DATA_FILTERS] = self.prefilters

        return yml_format
