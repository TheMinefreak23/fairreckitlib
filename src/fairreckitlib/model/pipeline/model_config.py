"""This module contains the model configuration.

Classes:

    ModelConfig: model configuration.

Functions:

    api_models_to_yml_format: format model configurations from different APIs to be yml compatible.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

from ...core.config.config_object import ObjectConfig


@dataclass
class ModelConfig(ObjectConfig):
    """Model Configuration.

    name: the name of the model.
    params: the parameters of the model.
    """
