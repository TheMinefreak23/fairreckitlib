"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.data.split.factory import SPLIT_RANDOM
from fairreckitlib.events import config_event
from fairreckitlib.experiment.parsing import assertion
from fairreckitlib.pipelines.data.pipeline import DatasetConfig
from fairreckitlib.pipelines.data.pipeline import SplitConfig
from ..constants import EXP_KEY_DATASETS
from ..constants import EXP_KEY_DATASET_NAME
from ..constants import EXP_KEY_DATASET_SPLIT
from ..constants import EXP_KEY_DATASET_SPLIT_TEST_RATIO
from ..constants import EXP_KEY_DATASET_SPLIT_TYPE


def parse_data_config(experiment_config, data_registry, split_factory, event_dispatcher):
    """Parses all dataset configurations.

    Args:
        experiment_config(dict): the experiment's total configuration.
        data_registry(DataRegistry): the data registry containing the available datasets.
        split_factory(dict): the split factory containing available splitters.
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
        split_factory(dict): the split factory containing available splitters.
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
        data_registry.get_available(),
        event_dispatcher,
        'PARSE ERROR: unknown dataset name \'' + str(dataset_name) + '\''
    ): return None, dataset_name

    # TODO parse these
    dataset_prefilters = []
    dataset_rating_modifier = None

    # attempt to parse dataset split
    dataset_splitting = parse_data_split_config(
        dataset_config,
        dataset_name,
        split_factory,
        event_dispatcher
    )
    # return on parse failure
    if dataset_splitting is None:
        return None, dataset_name

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
        split_factory(dict): the split factory containing available splitters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(SplitConfig): the parsed configuration or None on failure.
    """
    # assert EXP_KEY_DATASET_SPLIT is present
    if not assertion.is_key_in_dict(
        EXP_KEY_DATASET_SPLIT,
        dataset_config,
        event_dispatcher,
        'PARSE ERROR: dataset ' + dataset_name + ' missing key \'' +
        EXP_KEY_DATASET_SPLIT + '\' (required)'
    ): return None

    split_config = dataset_config[EXP_KEY_DATASET_SPLIT]

    # assert split_config is a dict
    if not assertion.is_type(
        split_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: dataset ' + dataset_name + ' invalid splitting value'
    ): return None

    # attempt to parse splitting test ratio
    test_ratio = parse_data_split_test_ratio(
        split_config,
        dataset_name,
        event_dispatcher
    )
    # return on parse failure
    if test_ratio is None:
        return None

    # parse splitting type
    split_type = parse_data_split_type(
        split_config,
        dataset_name,
        split_factory,
        event_dispatcher
    )

    parsed_config = SplitConfig(
        test_ratio,
        split_type
    )

    return parsed_config


def parse_data_split_test_ratio(split_config, dataset_name, event_dispatcher):
    """Parses a dataset splitting test ratio.

    Args:
        split_config(dict): the dataset's total configuration.
        dataset_name(str): the dataset name related to the splitting configuration.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        test_ratio(float): the parsed test ratio or None on failure.
    """
    # assert dataset splitting test ratio is present
    if not assertion.is_key_in_dict(
        EXP_KEY_DATASET_SPLIT_TEST_RATIO,
        split_config,
        event_dispatcher,
        'PARSE ERROR: dataset ' + dataset_name + ' missing splitting key \'' +
        EXP_KEY_DATASET_SPLIT_TEST_RATIO + '\' (required)'
    ): return None

    test_ratio = split_config[EXP_KEY_DATASET_SPLIT_TEST_RATIO]

    # assert test_ratio is a float
    if not assertion.is_type(
        test_ratio,
        float,
        event_dispatcher,
        'PARSE ERROR: dataset ' + dataset_name + ' invalid value for splitting key \'' +
        EXP_KEY_DATASET_SPLIT_TEST_RATIO + '\''
    ): return None

    # verify test_ratio is greater than zero
    if test_ratio <= 0.0:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg='PARSE ERROR: dataset ' + dataset_name + ' invalid splitting ' +
                EXP_KEY_DATASET_SPLIT_TEST_RATIO + ' \'' + str(test_ratio) + '\'' +
                '\n\t' + EXP_KEY_DATASET_SPLIT_TEST_RATIO + ' must be greater than zero'
        )
        return None

    # verify test_ratio is less than one
    if test_ratio >= 1.0:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg='PARSE ERROR: dataset ' + dataset_name + ' invalid splitting ' +
                EXP_KEY_DATASET_SPLIT_TEST_RATIO + ' \'' + str(test_ratio) + '\'' +
                '\n\t' + EXP_KEY_DATASET_SPLIT_TEST_RATIO + ' must be less than one'
        )
        return None

    return test_ratio


def parse_data_split_type(split_config, dataset_name, split_factory, event_dispatcher):
    """Parses a dataset splitting test ratio.

    Args:
        split_config(dict): the dataset's total configuration.
        dataset_name(str): the dataset name related to the splitting configuration.
        split_factory(dict): the split factory containing available splitters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        split_type(str): the parsed type or SPLIT_RANDOM on failure.
    """
    available_splits = split_factory.get_available_split_names()

    # assert dataset splitting type is present
    if not assertion.is_key_in_dict(
        EXP_KEY_DATASET_SPLIT_TYPE,
        split_config,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset_name + ' missing splitting key \'' +
        EXP_KEY_DATASET_SPLIT_TYPE + '\'',
        one_of_list=available_splits,
        default=SPLIT_RANDOM
    ): return SPLIT_RANDOM

    split_type = split_config[EXP_KEY_DATASET_SPLIT_TYPE]

    # assert split_type is available in the split factory
    if not assertion.is_one_of_list(
        split_type,
        available_splits,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset_name + ' invalid splitting ' +
        EXP_KEY_DATASET_SPLIT_TYPE + ' \'' + str(split_type) + '\'',
        default=SPLIT_RANDOM
    ): return SPLIT_RANDOM

    return split_type
