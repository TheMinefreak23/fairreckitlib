"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict

from ...core.config_constants import KEY_NAME, KEY_PARAMS
from .split_constants import KEY_SPLIT_TEST_RATIO


@dataclass
class SplitConfig:
    """Dataset Splitting Configuration."""

    name: str
    test_ratio: float
    params: Dict[str, Any]


def split_config_to_dict(split_config: SplitConfig) -> Dict[str, Any]:
    """Convert a splitting configuration to a dictionary.

    Args:
        split_config: the configuration to convert to dictionary.

    Returns:
        a dictionary containing the splitting configuration.
    """
    splitting = {
        KEY_NAME: split_config.name,
        KEY_SPLIT_TEST_RATIO: split_config.test_ratio
    }

    # only include splitting params if it has entries
    if len(split_config.params) > 0:
        splitting[KEY_PARAMS] = split_config.params

    return splitting
