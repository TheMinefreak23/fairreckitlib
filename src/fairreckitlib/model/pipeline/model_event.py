"""This module contains all callback functions used in the model pipeline.

Functions:

    get_model_events: get model pipeline events.
    on_begin_load_test_set: call when a test set is being loaded.
    on_begin_load_train_set: call when a train set is being loaded.
    on_begin_model_pipeline: call when the pipeline starts.
    on_begin_test_model: call when testing a model started.
    on_begin_train_model: call when training a model started.
    on_begin_model: call when a model computation started.
    on_end_load_test_set: call when a test set has been loaded.
    on_end_load_train_set: call when a train set has been loaded.
    on_end_model_pipeline: call when the pipeline ends.
    on_end_test_model: call when test a model finishes.
    on_end_train_model: call when training a model finishes.
    on_end_model: call when a model computation finishes.
    on_save_model_settings: call when the model settings have been saved.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, List, Tuple

ON_BEGIN_LOAD_TEST_SET = 'ModelPipeline.on_begin_load_test_set'
ON_BEGIN_LOAD_TRAIN_SET = 'ModelPipeline.on_begin_load_train_set'
ON_BEGIN_MODEL_PIPELINE = 'ModelPipeline.on_begin'
ON_BEGIN_TEST_MODEL = 'ModelPipeline.on_begin_model_test'
ON_BEGIN_TRAIN_MODEL = 'ModelPipeline.on_begin_model_train'
ON_BEGIN_MODEL = 'ModelPipeline.on_begin_model'
ON_END_LOAD_TEST_SET = 'ModelPipeline.on_end_load_test_set'
ON_END_LOAD_TRAIN_SET = 'ModelPipeline.on_end_load_train_set'
ON_END_MODEL_PIPELINE = 'ModelPipeline.on_end'
ON_END_TEST_MODEL = 'ModelPipeline.on_end_test_model'
ON_END_TRAIN_MODEL = 'ModelPipeline.on_end_train_model'
ON_END_MODEL = 'ModelPipeline.on_end_model'
ON_SAVE_MODEL_SETTINGS = 'ModelPipeline.on_save_model_settings'


def get_model_events() -> List[Tuple[str, Callable[[Any], None]]]:
    """Get all model pipeline events.

    The Call backs are specified below and serve as a default
    implementation for the RecommenderSystem class including the keyword arguments
    that are passed down by the model pipeline.

    Returns:
        a list of pairs in the format (event_id, func_on_event)
    """
    return [
        (ON_BEGIN_LOAD_TEST_SET, on_begin_load_test_set),
        (ON_BEGIN_LOAD_TRAIN_SET, on_begin_load_train_set),
        (ON_BEGIN_MODEL_PIPELINE, on_begin_model_pipeline),
        (ON_BEGIN_TEST_MODEL, on_begin_test_model),
        (ON_BEGIN_TRAIN_MODEL, on_begin_train_model),
        (ON_BEGIN_MODEL, on_begin_model),
        (ON_END_LOAD_TEST_SET, on_end_load_test_set),
        (ON_END_LOAD_TRAIN_SET, on_end_load_train_set),
        (ON_END_MODEL_PIPELINE, on_end_model_pipeline),
        (ON_END_TEST_MODEL, on_end_test_model),
        (ON_END_TRAIN_MODEL, on_end_train_model),
        (ON_END_MODEL, on_end_model),
        (ON_SAVE_MODEL_SETTINGS, on_save_model_settings)
    ]


def on_begin_load_test_set(event_listener: Any, **kwargs) -> None:
    """Call back when test set loading started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        test_set_path(str): the path where the test set is loaded from.
    """
    if event_listener.verbose:
        print('Loading test set from', kwargs['test_set_path'])


def on_begin_load_train_set(event_listener: Any, **kwargs) -> None:
    """Call back when train set loading started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        train_set_path(str): the path where the train set is loaded from.
    """
    if event_listener.verbose:
        print('Loading train set from', kwargs['train_set_path'])


def on_begin_model(event_listener: Any, **kwargs) -> None:
    """Call back when a model computation started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model_name(str): name of the model (algorithm).
    """
    if event_listener.verbose:
        print('Starting model:', kwargs['model_name'])


def on_begin_model_pipeline(event_listener: Any, **kwargs) -> None:
    """Call back when the model pipeline started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        api_name(str): the model API that will run in the pipeline.
        num_models(int): the number of models that will be processed.
    """
    if event_listener.verbose:
        print('\nStarting Model Pipeline:', kwargs['api_name'],
              'to process', kwargs['num_models'], 'model(s)')


def on_begin_test_model(event_listener: Any, **kwargs) -> None:
    """Call back when testing a model started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model(Algorithm): the model that will be tested.
        test_set(pandas.DataFrame): the test set that is used.
    """
    if event_listener.verbose:
        print('Testing model:', kwargs['model'].get_name())


def on_begin_train_model(event_listener: Any, **kwargs) -> None:
    """Call back when training a model started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model(Algorithm): the model that will be trained.
        train_set(pandas.DataFrame): the train set that is used.
    """
    if event_listener.verbose:
        print('Training model:', kwargs['model'].get_name())


def on_end_load_test_set(event_listener: Any, **kwargs) -> None:
    """Call back when test set loading finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        test_set_path(str): the path where the test set was loaded from.
        test_set(pandas.DataFrame): the loaded test set.
        elapsed_time(float): the time that has passed since the loading started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Loaded test set in {elapsed_time:1.4f}s')


def on_end_load_train_set(event_listener: Any, **kwargs) -> None:
    """Call back when train set loading finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        train_set_path(str): the path where the train set was loaded from.
        train_set(pandas.DataFrame): the loaded train set.
        elapsed_time(float): the time that has passed since the loading started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Loaded train set in {elapsed_time:1.4f}s')


def on_end_model(event_listener: Any, **kwargs) -> None:
    """Call back when a model computation finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model(Algorithm): the model that was used in the computation.
        elapsed_time(float): the time that has passed since the model
            computation started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print('Finished model:', kwargs['model'].get_name(), f'in {elapsed_time:1.4f}s')


def on_end_model_pipeline(event_listener: Any, **kwargs) -> None:
    """Call back when the model pipeline finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        api_name(str): the model API that completed the pipeline.
        num_models(int): the number of models that were processed.
        elapsed_time(float): the time that has passed since the pipeline
            started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print('Finished Model Pipeline:', kwargs['api_name'],
              f'in {elapsed_time:1.4f}s')


def on_end_test_model(event_listener: Any, **kwargs) -> None:
    """Call back when testing a model finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model(Algorithm): the model that was tested.
        test_set(pandas.DataFrame): the test set that was used.
        elapsed_time(float): the time that has passed since testing
            started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Tested model in {elapsed_time:1.4f}s')


def on_end_train_model(event_listener: Any, **kwargs) -> None:
    """Call back when training a model finished.

    Args:
        event_listener(): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model(Algorithm): the model that was trained.
        train_set(pandas.DataFrame): the train set that was used.
        elapsed_time(float): the time that has passed since training
            started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Trained model in {elapsed_time:1.4f}s')


def on_save_model_settings(event_listener: Any, **kwargs) -> None:
    """Call back when a model's settings are saved.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        settings_path(str): the path where the model parameter settings are saved.
        model_params(dict): dictionary containing the model parameter name-value pairs.
    """
    if event_listener.verbose:
        print('Saved model settings to:', kwargs['settings_path'])
