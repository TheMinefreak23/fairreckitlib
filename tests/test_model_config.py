"""This module tests the formatting and parsing of the model configuration of the experiment.

Functions:

    create_api_models_config: create api models config dictionary with all available algorithms.
    test_parse_models: test parsing the model configuration from the experiment configuration.
    test_parse_models_config: test parsing api-model pairs from the model configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List

import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.config.config_yml import format_yml_config_dict_list
from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS, VALID_TYPES
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.model.model_factory import KEY_MODELS, create_model_factory
from src.fairreckitlib.model.pipeline.model_config import ModelConfig
from src.fairreckitlib.model.pipeline.model_config_parsing import \
    parse_api_models_config, parse_models_config


model_factory = create_model_factory()

INVALID_CONTAINER_TYPES = \
    [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]


def create_api_models_config(model_type_factory: GroupFactory) -> Dict[str, List[Dict[str, Any]]]:
    """Create api models config dictionary with all available algorithms of the model type."""
    models_config = {}
    for api_name in model_type_factory.get_available_names():
        api_factory = model_type_factory.get_factory(api_name)
        models_config[api_name] = []
        for algo_name in api_factory.get_available_names():
            models_config[api_name].append({
                KEY_NAME: algo_name,
                KEY_PARAMS: api_factory.create_params(algo_name).get_defaults()
            })

    return models_config


@pytest.mark.parametrize('model_type', VALID_TYPES)
def test_parse_models_config(model_type: str, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing the model configuration from the experiment configuration."""
    model_type_factory = model_factory.get_factory(model_type)

    assert parse_models_config({}, model_type_factory, parse_event_dispatcher) is None, \
        'expected parsing failure for an empty experiment configuration'

    models = {KEY_MODELS: create_api_models_config(model_type_factory)}
    parsed_models = parse_models_config(models, model_type_factory, parse_event_dispatcher)

    assert len(models[KEY_MODELS]) == len(parsed_models), \
        'expected parsing to succeed for all models in the experiment configuration'


@pytest.mark.parametrize('model_type', VALID_TYPES)
def test_parse_api_models_config(model_type: str, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing api-model pairs from the model configuration for both model types."""
    model_type_factory = model_factory.get_factory(model_type)

    # test failure for parsing various types, including a dict that is empty
    for models_config in INVALID_CONTAINER_TYPES:
        parsed_models = parse_api_models_config(
            models_config,
            model_type_factory,
            parse_event_dispatcher
        )

        assert parsed_models is None, \
            'did not expect any parsed models for invalid models configuration'

    # test failure for unknown algorithm API name
    parsed_models = parse_api_models_config(
        {'unknown': []},
        model_type_factory,
        parse_event_dispatcher
    )
    assert parsed_models is None, \
        'did not expect any parsed models for unknown algorithm api'

    models_config = create_api_models_config(model_type_factory)

    # test parsing success for the entire configuration, including the formatted parse result
    parsed_models = parse_api_models_config(
        models_config,
        model_type_factory,
        parse_event_dispatcher
    )
    formatted_models_config = format_yml_config_dict_list(parsed_models)

    assert len(formatted_models_config) == len(models_config), \
        'expected all apis to be formatted from the models configuration'
    assert len(parsed_models) == len(models_config), \
        'expected all apis to be parsed from the models configuration'

    for parsed_api_name, parsed_api_models in parsed_models.items():
        assert parsed_api_name in formatted_models_config, \
            'expected parsed api name to be present in the formatted model configuration'
        assert parsed_api_name in models_config, \
            'expected parsed api name to be present in the original model configuration'

        api_config = models_config[parsed_api_name]
        formatted_api_config = formatted_models_config[parsed_api_name]
        assert len(formatted_api_config) == len(api_config), \
            'expected all api algorithms to be present in the formatted api configuration'
        assert len(parsed_api_models) == len(api_config), \
            'expected all api algorithms to be parsed from the original api configuration'

        for parsed_model in parsed_api_models:
            assert isinstance(parsed_model, ModelConfig), \
                'expected parsed model to be a ModelConfig'
            assert any(parsed_model.name is algo[KEY_NAME] for algo in formatted_api_config), \
                'expected parsed model to be present in the formatted api configuration'
            assert any(parsed_model.name is algo[KEY_NAME] for algo in api_config), \
                'expected parsed model to be present in the original api configuration'
