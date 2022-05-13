"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Dict, Any, List
from ...core.config_constants import KEY_NAME, KEY_PARAMS
from ...core.config_params import ConfigOptionParam, ConfigValueParam
from ...core.event_dispatcher import EventDispatcher
from ...core.factories import Factory, GroupFactory
from ...core.parsing.parse_assert import assert_is_type, assert_is_container_not_empty
from ...core.parsing.parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from ...core.parsing.parse_event import ON_PARSE
from ...core.parsing.parse_params import parse_config_param, parse_config_parameters
from ..set.dataset_registry import DataRegistry
from ..split.split_factory import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from ..split.split_factory import DEFAULT_SPLIT_TEST_RATIO, DEFAULT_SPLIT_TYPE
from .data_config import DatasetConfig, SplitConfig, KEY_DATASETS

def parse_data_config(experiment_config: Dict[str, Any], data_registry: DataRegistry,
                      data_factory: GroupFactory, event_dispatcher: EventDispatcher) \
                      -> List[DatasetConfig]:
    """Parse all dataset configurations.

    Args:
        experiment_config: the experiment's total configuration.
        data_registry: the data registry containing the available datasets.
        data_factory: #TODO explain this
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        parsed_config: list of parsed DatasetConfig's.
    """
    parsed_config = []

    # assert KEY_DATASETS is present
    if not assert_is_key_in_dict(
        KEY_DATASETS,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + KEY_DATASETS + '\' (required)',
        default=parsed_config
    ): return parsed_config

    datasets_config = experiment_config[KEY_DATASETS]

    # assert datasets_config is a list
    if not assert_is_type(
        datasets_config,
        list,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + KEY_DATASETS + '\'',
        default=parsed_config
    ): return parsed_config

    # assert datasets_config has list entries
    if not assert_is_container_not_empty(
        datasets_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + KEY_DATASETS + '\' is empty',
        default=parsed_config
    ): return parsed_config

    # parse datasets_config list entries
    for _, dataset_config in enumerate(datasets_config):
        dataset, dataset_name = parse_dataset_config(
            dataset_config,
            data_registry,
            data_factory,
            event_dispatcher
        )
        # skip on failure
        if dataset is None:
            event_dispatcher.dispatch(
                ON_PARSE,
                msg='PARSE WARNING: failed to parse dataset \'' +
                str(dataset_name) + '\', skipping...'
            )
            continue

        parsed_config.append(dataset)

    return parsed_config


def parse_dataset_config(dataset_config: Dict[str, Any], data_registry: DataRegistry,
                         data_factory: Factory, event_dispatcher: EventDispatcher) \
                         -> tuple[DatasetConfig, str]:
    """Parse a dataset configuration.

    Args:
        dataset_config: the dataset's configuration.
        data_registry: the data registry containing the available datasets.
        data_factory: #TODO explain this
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        parsed_config: the parsed configuration or None on failure.
        dataset_name: the name of the parsed dataset or None on failure.
    """
    # assert dataset_config is a dict
    if not assert_is_type(
        dataset_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid dataset entry'
    ): return None, None

    # assert dataset name is present
    if not assert_is_key_in_dict(
        KEY_NAME,
        dataset_config,
        event_dispatcher,
        'PARSE ERROR: missing dataset key \'' + KEY_NAME + '\' (required)'
    ): return None, None

    dataset_name = dataset_config[KEY_NAME]

    # assert dataset name is available in the data registry
    if not assert_is_one_of_list(
        dataset_name,
        data_registry.get_available_sets(),
        event_dispatcher,
        'PARSE ERROR: unknown dataset name \'' + str(dataset_name) + '\''
    ): return None, dataset_name

    # TODO parse these
    dataset_prefilters = []
    dataset_rating_modifier = None

    # parse dataset split
    dataset_splitting = parse_data_split_config(
        dataset_config,
        dataset_name,
        data_factory.get_factory(KEY_SPLITTING),
        event_dispatcher
    )

    parsed_config = DatasetConfig(
        dataset_name,
        dataset_prefilters,
        dataset_rating_modifier,
        dataset_splitting
    )

    return parsed_config, dataset_name


def parse_data_split_config(dataset_config: Dict[str, Any], dataset_name: str,
                            split_factory: Factory, event_dispatcher: EventDispatcher) \
                            -> SplitConfig:
    """Parse a dataset splitting configuration.

    Args:
        dataset_config: the dataset's total configuration.
        dataset_name: the dataset name related to the splitting configuration.
        split_factory: the split factory containing available splitters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        parsed_config: the parsed configuration or None on failure.
    """
    parsed_config = SplitConfig(
        DEFAULT_SPLIT_TEST_RATIO,
        DEFAULT_SPLIT_TYPE,
        split_factory.create_params(DEFAULT_SPLIT_TYPE).get_defaults()
    )

    # dataset splitting is optional
    if KEY_SPLITTING not in dataset_config:
        event_dispatcher.dispatch(
            ON_PARSE,
            msg='PARSE WARNING: dataset ' + dataset_name + ' missing key \'' +
                KEY_SPLITTING + '\'',
            default=parsed_config
        )
        return parsed_config

    split_config = dataset_config[KEY_SPLITTING]

    # assert split_config is a dict
    if not assert_is_type(
        split_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset_name + ' invalid splitting value',
        default=parsed_config
    ): return parsed_config

    # parse splitting test ratio
    success, test_ratio = parse_config_param(
        split_config,
        dataset_name + ' ' + KEY_SPLITTING,
        ConfigValueParam(
            KEY_SPLIT_TEST_RATIO,
            float,
            DEFAULT_SPLIT_TEST_RATIO,
            (0.01, 0.99)
        ),
        event_dispatcher
    )
    if success:
        parsed_config.test_ratio = test_ratio

    # parse splitting type
    success, split_type = parse_config_param(
        split_config,
        dataset_name + ' ' + KEY_SPLITTING,
        ConfigOptionParam(
            KEY_NAME,
            str,
            DEFAULT_SPLIT_TYPE,
            split_factory.get_available_names()
        ),
        event_dispatcher
    )
    if success:
        parsed_config.type = split_type

    split_params = split_factory.create_params(split_type)
    parsed_config.params = split_params.get_defaults()

    # assert KEY_PARAMS is present
    # skip when the splitter has no parameters at all
    if split_params.get_num_params() > 0 and assert_is_key_in_dict(
        KEY_PARAMS,
        split_config,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset_name + ' ' + KEY_SPLITTING + ' missing key \'' +
        KEY_PARAMS + '\'',
        default=parsed_config.params
    ):
        # parse the splitter parameters
        parsed_config.params = parse_config_parameters(
            split_config[KEY_PARAMS],
            dataset_name,
            split_params,
            event_dispatcher
        )

    return parsed_config
