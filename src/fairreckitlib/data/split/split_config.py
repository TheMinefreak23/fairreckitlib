"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict

from ...core.config_object import ObjectConfig
from .split_constants import KEY_SPLIT_TEST_RATIO


@dataclass
class SplitConfig(ObjectConfig):
    """Dataset Splitting Configuration."""

    test_ratio: float

    def to_yml_format(self) -> Dict[str, Any]:
        """Format splitting configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the splitting configuration.
        """
        yml_format = ObjectConfig.to_yml_format(self)
        yml_format[KEY_SPLIT_TEST_RATIO] = self.test_ratio
        return yml_format
