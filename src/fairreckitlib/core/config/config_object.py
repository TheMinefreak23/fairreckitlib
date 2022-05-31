"""This module contains the base functionality for an object's configuration.

Classes:

    ObjectConfig: base class configuration for an object with a name and parameters.

Functions:

    object_config_list_to_yml_format: convert list of ObjectConfig's to a yml format.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..core_constants import KEY_NAME, KEY_PARAMS
from .config_yml import YmlConfig


@dataclass
class ObjectConfig(YmlConfig):
    """Base Object Configuration.

    name: the name of the object.
    params: the parameters of the object.
    """

    name: str
    params: Dict[str, Any]

    def to_yml_format(self) -> Dict[str, Any]:
        """Format object configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the object configuration.
        """
        yml_format = { KEY_NAME: self.name }

        # only include object params if it has entries
        if len(self.params) > 0:
            yml_format[KEY_PARAMS] = dict(self.params)

        return yml_format
