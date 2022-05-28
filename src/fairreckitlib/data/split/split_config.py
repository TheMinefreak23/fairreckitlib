"""This module contains the split configuration.

Classes:

    SplitConfig: split configuration.

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
    """Dataset Splitting Configuration.

    name: the name of the splitter.
    params: the parameters of the splitter.
    test_ratio: the test ratio used by the splitter.
    """

    test_ratio: float

    def get_split_ratio_string(self) -> str:
        """Get the split ratio percentages formatted as a string.

        Returns:
            a string containing the split ratio in percentages.
        """
        test_perc = int(self.test_ratio * 100.0)
        return str(100 - test_perc) + '/' + str(test_perc)

    def to_yml_format(self) -> Dict[str, Any]:
        """Format splitting configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the splitting configuration.
        """
        yml_format = ObjectConfig.to_yml_format(self)
        yml_format[KEY_SPLIT_TEST_RATIO] = self.test_ratio
        return yml_format
