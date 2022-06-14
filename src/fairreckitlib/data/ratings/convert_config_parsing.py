"""This module contains a parser for the dataset rating conversion configuration.

Functions:

    parse_data_convert_config: parse convert configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Optional

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_config_object import parse_config_object
from ..set.dataset import Dataset
from .convert_config import ConvertConfig
from .convert_constants import KEY_RATING_CONVERTER


def parse_data_convert_config(
        dataset_config: Dict[str, Any],
        dataset: Dataset,
        matrix_name: str,
        converter_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Optional[ConvertConfig]:
    """Parse a dataset rating converter configuration.

    Args:
        dataset_config: the dataset's total configuration.
        dataset: the dataset related to the converter configuration.
        matrix_name: the dataset's matrix name to use.
        converter_factory: the converter factory containing available converters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration or None on failure.
    """
    # dataset rating conversion is optional
    if KEY_RATING_CONVERTER not in dataset_config:
        return None

    dataset_converter_factory = converter_factory.get_factory(dataset.get_name())
    matrix_converter_factory = dataset_converter_factory.get_factory(matrix_name)

    converter, _ = parse_config_object(
        'dataset ' + dataset.get_name() + ' \'' + matrix_name + '\' rating converter',
        dataset_config[KEY_RATING_CONVERTER],
        matrix_converter_factory,
        event_dispatcher
    )

    return ConvertConfig(converter.name, converter.params) if bool(converter) else None
