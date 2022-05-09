"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.events import config_event
from fairreckitlib.experiment.parsing import assertion
from fairreckitlib.pipelines.data.pipeline import DatasetConfig
from fairreckitlib.pipelines.data.pipeline import SplitConfig
from ..constants import EXP_DEFAULT_SPLIT_TEST_RATIO
from ..constants import EXP_DEFAULT_SPLIT_TYPE
from ..constants import EXP_KEY_DATASETS
from ..constants import EXP_KEY_DATASET_NAME
from ..constants import EXP_KEY_DATASET_SPLIT
from ..constants import EXP_KEY_DATASET_SPLIT_PARAMS
from ..constants import EXP_KEY_DATASET_SPLIT_TEST_RATIO
from ..constants import EXP_KEY_DATASET_SPLIT_TYPE
from ..params import ConfigOptionParam
from ..params import ConfigValueParam
from ..parsing.params import parse_config_param
from ..parsing.params import parse_config_parameters


def parse_data_config(experiment_config, data_registry, split_factory, event_dispatcher):
    """Parses all dataset configurations.

    Args:
        experiment_config(dict): the experiment's total configuration.
        data_registry(DataRegistry): the data registry containing the available datasets.
        split_factory(SplitFactory): the split factory containing available splitters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(array like): list of parsed DatasetConfig's.
    """
    parsed_config = []

    # assert EXP_KEY_DATASETS is present
    if not assertion.is_key_in_dict(
        EXP_KEY_DATASETS,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + EXP_KEY_DATASETS + '\' (required)',
        default=parsed_config
    ): return parsed_config

    datasets_config = experiment_config[EXP_KEY_DATASETS]

    # assert datasets_config is a list
    if not assertion.is_type(
        datasets_config,
        list,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + EXP_KEY_DATASETS + '\'',
        default=parsed_config
    ): return parsed_config

    # assert datasets_config has list entries
    if not assertion.is_container_not_empty(
        datasets_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + EXP_KEY_DATASETS + '\' is empty',
        default=parsed_config
    ): return parsed_config

    # parse datasets_config list entries
    for _, dataset_config in enumerate(datasets_config):
        dataset, dataset_name = parse_dataset_config(
            dataset_config,
            data_registry,
            split_factory,
            event_dispatcher
        )
        # skip on failure
        if dataset is None:
            event_dispatcher.dispatch(
                config_event.ON_PARSE,
                msg='PARSE WARNING: failed to parse dataset \'' +
                str(dataset_name) + '\', skipping...'
            )
            continue

        parsed_config.append(dataset)

    return parsed_config


def parse_dataset_config(dataset_config, data_registry, split_factory, event_dispatcher):
    """Parses a dataset configuration.

    Args:
        dataset_config(dict): the dataset's configuration.
        data_registry(DataRegistry): the data registry containing the available datasets.
        split_factory(SplitFactory): the split factory containing available splitters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(DatasetConfig): the parsed configuration or None on failure.
        dataset_name(str): the name of the parsed dataset or None on failure.
    """
    # assert dataset_config is a dict
    if not assertion.is_type(
        dataset_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid dataset entry'
    ): return None, None

    # assert dataset name is present
    if not assertion.is_key_in_dict(
        EXP_KEY_DATASET_NAME,
        dataset_config,
        event_dispatcher,
        'PARSE ERROR: missing dataset key \'' + EXP_KEY_DATASET_NAME + '\' (required)'
    ): return None, None

    dataset_name = dataset_config[EXP_KEY_DATASET_NAME]

    # assert dataset name is available in the data registry
    if not assertion.is_one_of_list(
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
        split_factory,
        event_dispatcher
    )

    parsed_config = DatasetConfig(
        dataset_name,
        dataset_prefilters,
        dataset_rating_modifier,
        dataset_splitting
    )

    return parsed_config, dataset_name


def parse_data_split_config(dataset_config, dataset_name, split_factory, event_dispatcher):
    """Parses a dataset splitting configuration.

    Args:
        dataset_config(dict): the dataset's total configuration.
        dataset_name(str): the dataset name related to the splitting configuration.
        split_factory(SplitFactory): the split factory containing available splitters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(SplitConfig): the parsed configuration or None on failure.
    """
    parsed_config = SplitConfig(
        EXP_DEFAULT_SPLIT_TEST_RATIO,
        EXP_DEFAULT_SPLIT_TYPE,
        split_factory.get_split_params(EXP_DEFAULT_SPLIT_TYPE).get_defaults()
    )

    # dataset splitting is optional
    if EXP_KEY_DATASET_SPLIT not in dataset_config:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg='PARSE WARNING: dataset ' + dataset_name + ' missing key \'' +
                EXP_KEY_DATASET_SPLIT + '\'',
            default=parsed_config
        )
        return parsed_config

    split_config = dataset_config[EXP_KEY_DATASET_SPLIT]

    # assert split_config is a dict
    if not assertion.is_type(
        split_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset_name + ' invalid splitting value',
        default=parsed_config
    ): return parsed_config

    # parse splitting test ratio
    success, test_ratio = parse_config_param(
        split_config,
        dataset_name,
        ConfigValueParam(
            EXP_KEY_DATASET_SPLIT_TEST_RATIO,
            float,
            EXP_DEFAULT_SPLIT_TEST_RATIO,
            (0.01, 0.99)
        ),
        event_dispatcher
    )
    if success:
        parsed_config.test_ratio = test_ratio

    # parse splitting type
    success, split_type = parse_config_param(
        split_config,
        dataset_name,
        ConfigOptionParam(
            EXP_KEY_DATASET_SPLIT_TYPE,
            str,
            EXP_DEFAULT_SPLIT_TYPE,
            split_factory.get_available_split_names()
        ),
        event_dispatcher
    )
    if success:
        parsed_config.type = split_type

    split_params = split_factory.get_split_params(split_type)
    parsed_config.params = split_params.get_defaults()

    # assert EXP_KEY_DATASET_SPLIT_PARAMS is present
    # skip when the splitter has no parameters at all
    if split_params.get_num_params() > 0 and assertion.is_key_in_dict(
        EXP_KEY_DATASET_SPLIT_PARAMS,
        split_config,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset_name + ' missing key \'' +
        EXP_KEY_DATASET_SPLIT_PARAMS + '\'',
        default=parsed_config.params
    ):
        # parse the splitter parameters
        parsed_config.params = parse_config_parameters(
            split_config[EXP_KEY_DATASET_SPLIT_PARAMS],
            dataset_name,
            split_params,
            event_dispatcher
        )

    return parsed_config
