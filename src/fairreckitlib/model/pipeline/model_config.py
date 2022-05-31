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
from typing import Any, Dict, List

from ...core.config.config_object import ObjectConfig
from ...core.config.config_yml import format_yml_config_list


@dataclass
class ModelConfig(ObjectConfig):
    """Model Configuration.

    name: the name of the model.
    params: the parameters of the model.
    """


def api_models_to_yml_format(
        api_models: Dict[str, List[ModelConfig]]) -> Dict[str, List[Dict[str, Any]]]:
    """Format API models configuration list to a yml compatible dictionary.

    Returns:
        a dictionary containing the lists of model configurations.
    """
    yml_format = {}

    for api_name, models in api_models.items():
        yml_format[api_name] = format_yml_config_list(models)

    return yml_format
