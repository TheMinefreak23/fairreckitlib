"""This module contains the convert configuration.

Classes:

    ConvertConfig: convert configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

from ...core.config_object import ObjectConfig


@dataclass
class ConvertConfig(ObjectConfig):
    """Dataset rating conversion Configuration.

    name: the name of the rating converter.
    params: the parameters of the rating converter.
    """
