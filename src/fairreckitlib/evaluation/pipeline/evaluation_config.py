"""This module contains the metric configuration.

Classes:

    MetricConfig: metric configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...core.config.config_object import ObjectConfig
from ...data.filter.filter_config import DataSubsetConfig
from ..metrics.metric_constants import KEY_METRIC_SUBGROUP


@dataclass
class MetricConfig(ObjectConfig):
    """Metric Configuration.

    name: the name of the metric.
    params: the parameters of the metric.
    subgroup: the subgroup of the metric.
    """

    subgroup: Optional[DataSubsetConfig]

    def to_yml_format(self) -> Dict[str, Any]:
        """Format metric configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the metric configuration.
        """
        yml_format = ObjectConfig.to_yml_format(self)
        # only include subgroup if it is specified
        if self.subgroup is not None:
            yml_format[KEY_METRIC_SUBGROUP] = self.subgroup.to_yml_format()

        return yml_format
