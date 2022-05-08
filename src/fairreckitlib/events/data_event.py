"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

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


def get_data_events():
    """Gets all data pipeline events.

    The callback functions are specified below and serve as a default
    implementation for the RecommenderSystem class including the keyword arguments
    that are passed down by the data pipeline.

    Returns:
        (array like) list of pairs in the format (event_id, func_on_event)
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


def on_begin_data_pipeline(event_listener, **kwargs):
    """Callback function when the data pipeline started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that is being processed by the pipeline.
    """
    if event_listener.verbose:
        print('\nStarting Data Pipeline:', kwargs['dataset'].name)


def on_begin_filter_dataset(event_listener, **kwargs):
    """Callback function when dataset filtering started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        prefilters(array like): list of filters that will be applied to the dataset.
    """
    if event_listener.verbose:
        print('Filtering dataset using:', kwargs['prefilters'])


def on_begin_load_dataset(event_listener, **kwargs):
    """Callback function when dataset loading started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that is being loaded.
    """
    if event_listener.verbose:
        print('Loading dataset from', kwargs['dataset'].get_matrix_file_path())


def on_begin_modify_dataset(event_listener, **kwargs):
    """Callback function when dataset rating modification started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        rating_modifier(object): the rating modifier object that will be
            applied to the dataset.
    """
    if event_listener.verbose:
        print('Modifying dataset ratings:', kwargs['rating_modifier'])


def on_begin_save_sets(event_listener, **kwargs):
    """Callback function when train/test set saving started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        train_set_path(str): the path where the test set is saved to.
        test_set_path(str): the path where the test set is saved to.
    """
    if event_listener.verbose:
        print('Saving train set to', kwargs['train_set_path'])
        print('Saving test set to', kwargs['test_set_path'])


def on_begin_split_dataset(event_listener, **kwargs):
    """Callback function when dataset splitting started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        split_type(str): the type of split that is performed.
        test_ratio(float): the ratio of the test set size relative
            to the original dataset.
    """
    if event_listener.verbose:
        test_perc = int(kwargs['test_ratio'] * 100.0)
        print('Splitting dataset:', 100 - test_perc, '/', test_perc,
              '=>', kwargs['split_type'])


def on_end_data_pipeline(event_listener, **kwargs):
    """Callback function when the data pipeline finished.

    Args:
        event_listener(object): the listener that is registered
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


def on_end_filter_dataset(event_listener, **kwargs):
    """Callback function when dataset filtering finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        prefilters(array like): list of filters that are applied to the dataset.
        elapsed_time(float): the time that has passed since the filtering started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Filtered dataset in {elapsed_time:1.4f}s')


def on_end_load_dataset(event_listener, **kwargs):
    """Callback function when dataset loading finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dataset(Dataset): the dataset that is loaded.
        elapsed_time(float): the time that has passed since the loading started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Loaded dataset in {elapsed_time:1.4f}s')


def on_end_modify_dataset(event_listener, **kwargs):
    """Callback function when dataset rating modification finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        rating_modifier(object): the rating modifier object that was
            applied to the dataset.
        elapsed_time(float): the time that has passed since the
            modification started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Modified dataset ratings in {elapsed_time:1.4f}s')


def on_end_save_sets(event_listener, **kwargs):
    """Callback function when train/test set saving finished.

    Args:
        event_listener(object): the listener that is registered
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


def on_end_split_dataset(event_listener, **kwargs):
    """Callback function when dataset splitting finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        split_type(str): the type of split that was performed.
        test_ratio(float): the ratio of the test set size relative
            to the original dataset.
        elapsed_time(float): the time that has passed since the
            splitting started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Split dataset in {elapsed_time:1.4f}s')
