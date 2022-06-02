"""This module contains the base functionality for an object's configuration.

Classes:

    YmlConfig: base class configuration that is compatible with an yml format.

Functions:

    format_yml_config_dict: convert dict of YmlConfig's to an yml format.
    format_yml_config_list: convert list of YmlConfig's to an yml format.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class YmlConfig(metaclass=ABCMeta):
    """Base YML Configuration."""

    @abstractmethod
    def to_yml_format(self) -> Dict[str, Any]:
        """Format configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the configuration.
        """
        raise NotImplementedError()


def format_yml_config_dict(yml_configs: Dict[str, YmlConfig]) -> Dict[str, Any]:
    """Format yml configuration dictionary.

    Returns:
        a list containing the yml configuration's.
    """
    yml_format = {}

    for name, config in yml_configs.items():
        yml_format[name] = config.to_yml_format()

    return yml_format


def format_yml_config_list(yml_configs: List[YmlConfig]) -> List[Dict[str, Any]]:
    """Format yml configuration list to a yml compatible list.

    Returns:
        a list containing the yml configuration's.
    """
    yml_format = []

    for config in yml_configs:
        yml_format.append(config.to_yml_format())

    return yml_format
