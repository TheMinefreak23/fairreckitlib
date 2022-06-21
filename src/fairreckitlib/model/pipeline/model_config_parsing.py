"""This module contains a parser for the model configuration(s).

Functions:

    parse_models_config: parse models from the experiment configuration.
    parse_api_models_config: parse multiple API model configurations.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_assert import assert_is_type, assert_is_container_not_empty
from ...core.parsing.parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from ...core.parsing.parse_config_object import parse_config_object_list
from ..model_factory import KEY_MODELS
from .model_config import ModelConfig


def parse_models_config(
        experiment_config: Dict[str, Any],
        model_type_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Optional[Dict[str, List[ModelConfig]]]:
    """Parse the experiment KEY_MODELS section of the experiment configuration.

    Args:
        experiment_config: the experiment's total configuration.
        model_type_factory: the model type factory containing the available models.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        a dictionary of parsed ModelConfig's keyed by API name or None when empty.
    """
    # assert EXP_KEY_MODELS is present
    if not assert_is_key_in_dict(
        KEY_MODELS,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + KEY_MODELS + '\' (required)'
    ): return None

    return parse_api_models_config(
        experiment_config[KEY_MODELS],
        model_type_factory,
        event_dispatcher
    )


def parse_api_models_config(
        models_config: Any,
        model_type_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Optional[Dict[str, List[ModelConfig]]]:
    """Parse all model configurations.

    Args:
        models_config: the KEY_MODELS configuration.
        model_type_factory: the model type factory containing the available models.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        a dictionary of parsed ModelConfig's keyed by API name or None when empty.
    """
    parsed_config = None

    # assert models_config is a dict
    if not assert_is_type(
        models_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + KEY_MODELS + '\'',
        default_value=parsed_config
    ): return parsed_config

    # assert models_config has entries
    if not assert_is_container_not_empty(
        models_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + KEY_MODELS + '\' is empty',
        default_value=parsed_config
    ): return parsed_config

    # parse models_config entries
    parsed_config = {}
    for api_name, models in models_config.items():
        # attempt to parse a list of ModelConfig's for this API
        if not assert_is_one_of_list(
            api_name,
            model_type_factory.get_available_names(),
            event_dispatcher,
            'PARSE WARNING: unknown model API \'' + api_name + '\''
        ): continue

        parsed_config_objs = parse_config_object_list(
            api_name,
            'model',
            models,
            model_type_factory.get_factory(api_name),
            event_dispatcher
        )

        # skip when no configurations are actually parsed
        if not assert_is_container_not_empty(
            parsed_config_objs,
            event_dispatcher,
            'PARSE WARNING: skipping models for API: ' + api_name
        ): continue

        parsed_config[api_name] = [ModelConfig(m.name, m.params) for (m, _) in parsed_config_objs]

    # final check to verify at least one model got parsed
    if not assert_is_container_not_empty(
        parsed_config,
        event_dispatcher,
        'PARSE ERROR: no experiment ' + KEY_MODELS + ' specified'
    ): return None

    return parsed_config
