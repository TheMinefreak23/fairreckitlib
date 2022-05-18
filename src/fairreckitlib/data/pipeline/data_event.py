"""This module contains all event ids and callback functions used in the data pipeline.

Constants:

    ON_BEGIN_DATA_PIPELINE: id of the event that is used when the data pipeline starts.
    ON_BEGIN_FILTER_DATASET: id of the event that is used when dataset filtering starts.
    ON_BEGIN_LOAD_DATASET: id of the event that is used when a dataset is being loaded.
    ON_BEGIN_MODIFY_DATASET: id of the event that is used when dataset ratings are being modified.
    ON_BEGIN_SAVE_SETS: id of the event that is used when the train and test sets are being saved.
    ON_BEGIN_SPLIT_DATASET: id of the event that is used when a dataset is being split.
    ON_END_DATA_PIPELINE: id of the event that is used when the data pipeline ends.
    ON_END_FILTER_DATASET: id of the event that is used when dataset filtering finishes.
    ON_END_LOAD_DATASET: id of the event that is used when a dataset has been loaded.
    ON_END_MODIFY_DATASET: id of the event that is used when dataset ratings have been modified.
    ON_END_SAVE_SETS: id of the event that is used when the train and test sets have been saved.
    ON_END_SPLIT_DATASET: id of the event that is used when a dataset has been split.

Functions:

    get_data_events: get data pipeline events.
    on_begin_data_pipeline: call when the data pipeline starts.
    on_begin_filter_dataset: call when dataset filtering starts.
    on_begin_load_dataset: call when a dataset is being loaded.
    on_begin_modify_dataset: call when dataset ratings are being modified.
    on_begin_save_set: call when the train and test sets are being saved.
    on_begin_split_dataset: call when a dataset is being split.
    on_end_data_pipeline: call when the data pipeline ends.
    on_end_filter_dataset: call when dataset filtering finishes.
    on_end_load_dataset: call when a dataset has been loaded.
    on_end_modify_dataset: call when dataset ratings have been modified.
    on_end_save_set: call when the train and test sets have been saved.
    on_end_split_dataset: call when a dataset has been split.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, List, Tuple

ON_BEGIN_DATA_PIPELINE = 'DataPipeline.on_begin'
ON_BEGIN_FILTER_DATASET = 'DataPipeline.on_begin_filter_dataset'
ON_BEGIN_LOAD_DATASET = 'DataPipeline.on_begin_load_dataset'
ON_BEGIN_MODIFY_DATASET = 'DataPipeline.on_begin_modify_dataset'
ON_BEGIN_SAVE_SETS = 'DataPipeline.on_begin_save_sets'
ON_BEGIN_SPLIT_DATASET = 'DataPipeline.on_begin_split_dataset'
ON_END_DATA_PIPELINE = 'DataPipeline.on_end'
ON_END_FILTER_DATASET = 'DataPipeline.on_end_filter_dataset'
ON_END_LOAD_DATASET = 'DataPipeline.on_end_load_dataset'
ON_END_MODIFY_DATASET = 'DataPipeline.on_end_modify_dataset'
ON_END_SAVE_SETS = 'DataPipeline.on_end_save_sets'
ON_END_SPLIT_DATASET = 'DataPipeline.on_end_split_dataset'


def get_data_events() -> List[Tuple[str, Callable[[Any], None]]]:
    """Get all data pipeline events.

    The callback functions are specified below and serve as a default
    implementation for the RecommenderSystem class including the keyword arguments
    that are passed down by the data pipeline.

    Returns:
        a list of pairs in the format (event_id, func_on_event)
    """
    return [
        (ON_BEGIN_DATA_PIPELINE, on_begin_data_pipeline),
        (ON_BEGIN_FILTER_DATASET, on_begin_filter_dataset),
        (ON_BEGIN_LOAD_DATASET, on_begin_load_dataset),
        (ON_BEGIN_MODIFY_DATASET, on_begin_modify_dataset),
        (ON_BEGIN_SAVE_SETS, on_begin_save_sets),
        (ON_BEGIN_SPLIT_DATASET, on_begin_split_dataset),
        (ON_END_DATA_PIPELINE, on_end_data_pipeline),
        (ON_END_FILTER_DATASET, on_end_filter_dataset),
        (ON_END_LOAD_DATASET, on_end_load_dataset),
        (ON_END_MODIFY_DATASET, on_end_modify_dataset),
        (ON_END_SAVE_SETS, on_end_save_sets),
        (ON_END_SPLIT_DATASET, on_end_split_dataset)
    ]


def on_begin_data_pipeline(event_listener: Any, **kwargs) -> None:
    """Call back when the data pipeline starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that is being processed by the pipeline.
    """
    if event_listener.verbose:
        print('\nStarting Data Pipeline:', kwargs['dataset'].name)


def on_begin_filter_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset filtering starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        prefilters(array like): list of filters that will be applied to the dataset.
    """
    if event_listener.verbose:
        print('Filtering dataset using:', kwargs['prefilters'])


def on_begin_load_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset loading starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that is being loaded.
    """
    if event_listener.verbose:
        print('Loading dataset from', kwargs['dataset'].get_matrix_file_path())


def on_begin_modify_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset rating conversion starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        rating_modifier(object): the rating modifier object that will be
            applied to the dataset.
    """
    if event_listener.verbose:
        print('Converting dataset ratings:', kwargs['rating_converter'])


def on_begin_save_sets(event_listener: Any, **kwargs) -> None:
    """Call back when train/test set saving starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        train_set_path(str): the path where the test set is saved to.
        test_set_path(str): the path where the test set is saved to.
    """
    if event_listener.verbose:
        print('Saving train set to', kwargs['train_set_path'])
        print('Saving test set to', kwargs['test_set_path'])


def on_begin_split_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset splitting starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        split_name(str): the name of the split that is performed.
        test_ratio(float): the ratio of the test set size relative
            to the original dataset.
    """
    if event_listener.verbose:
        test_perc = int(kwargs['test_ratio'] * 100.0)
        print('Splitting dataset:', 100 - test_perc, '/', test_perc,
              '=>', kwargs['split_name'])


def on_end_data_pipeline(event_listener: Any, **kwargs) -> None:
    """Call back when the data pipeline finishes.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that was processed by the pipeline.
        elapsed_time(float): the time that has passed since the pipeline started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print('Finished Data Pipeline:', kwargs['dataset'].name,
              f'in {elapsed_time:1.4f}s')


def on_end_filter_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset filtering finishes.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        prefilters(array like): list of filters that are applied to the dataset.
        elapsed_time(float): the time that has passed since the filtering started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Filtered dataset in {elapsed_time:1.4f}s')


def on_end_load_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset loading finishes.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that is loaded.
        elapsed_time(float): the time that has passed since the loading started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Loaded dataset in {elapsed_time:1.4f}s')


def on_end_modify_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset rating conversion finishes.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        rating_modifier(object): the rating modifier object that was
            applied to the dataset.
        elapsed_time(float): the time that has passed since the
            modification started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Converted dataset ratings in {elapsed_time:1.4f}s')


def on_end_save_sets(event_listener: Any, **kwargs) -> None:
    """Call back when train/test set saving finishes.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        train_set_path(str): the path where the test set is saved.
        test_set_path(str): the path where the test set is saved.
        elapsed_time(float): the time that has passed since the
            saving started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Saved train and test sets in {elapsed_time:1.4f}s')


def on_end_split_dataset(event_listener: Any, **kwargs) -> None:
    """Call back when dataset splitting finishes.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        split_name(str): the name of the split that was performed.
        test_ratio(float): the ratio of the test set size relative
            to the original dataset.
        elapsed_time(float): the time that has passed since the
            splitting started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Split dataset in {elapsed_time:1.4f}s')
