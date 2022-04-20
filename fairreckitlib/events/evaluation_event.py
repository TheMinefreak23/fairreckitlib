"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.events.model_event import *

ON_BEGIN_LOAD_TEST_SET = 'on_begin_load_test_set'
ON_END_LOAD_TEST_SET = 'on_end_load_test_set'
ON_BEGIN_LOAD_TRAIN_SET = 'on_begin_load_train_set'
ON_END_LOAD_TRAIN_SET = 'on_end_load_train_set'
ON_BEGIN_LOAD_RECS_SET = 'on_begin_load_recs_set'
ON_END_LOAD_RECS_SET = 'on_end_load_recs_set'
ON_BEGIN_EVAL_PIPELINE = 'on_begin_eval_pipeline'
ON_END_EVAL_PIPELINE = 'on_end_eval_pipeline'
ON_BEGIN_EVAL = 'on_begin_eval'
ON_END_EVAL = 'on_end_eval'
ON_BEGIN_FILTER = 'on_begin_filter'
ON_END_FILTER = 'on_end_filter'


def get_evaluation_events():
    return [
        (ON_BEGIN_LOAD_TEST_SET, on_begin_load_test_set),
        (ON_END_LOAD_TEST_SET, on_end_load_test_set),
        (ON_BEGIN_LOAD_TRAIN_SET, on_begin_load_train_set),
        (ON_END_LOAD_TRAIN_SET, on_end_load_train_set),
        (ON_BEGIN_LOAD_RECS_SET, on_begin_load_recs_set),
        (ON_END_LOAD_RECS_SET, on_end_load_recs_set),
        (ON_BEGIN_EVAL_PIPELINE, on_begin_eval_pipeline),
        (ON_END_EVAL_PIPELINE, on_end_eval_pipeline),
        (ON_BEGIN_EVAL, on_begin_eval),
        (ON_END_EVAL, on_end_eval),
        (ON_BEGIN_FILTER, on_begin_filter),
        (ON_END_FILTER, on_end_filter)
    ]


def on_begin_load_recs_set(event_listener, **kwargs):
    """Callback function when test set loading started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        recs_set_path(str): the path where the test set is loaded from.
    """
    if event_listener.verbose:
        print('Loading recs set from', kwargs['recs_set_path'])


def on_end_load_recs_set(event_listener, **kwargs):
    """Callback function when test set loading finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        recs_set_path(str): the path where the recs set was loaded from.
        recs_set(pandas.DataFrame): the loaded recs set.
        elapsed_time(float): the time that has passed since the loading started,
            expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Loaded recs set in {elapsed_time:1.4f}s')


def on_begin_eval(event_listener, **kwargs):
    """Callback function when a model computation started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        metric(Metric): metric
    """
    if event_listener.verbose:
        print('Starting evaluation with metric', kwargs['metric'].value)


def on_end_eval(event_listener, **kwargs):
    """Callback function when a model computation finished.

        Args:
            event_listener(object): the listener that is registered
                in the event dispatcher with this callback.

        Keyword Args:
            metric(Metric): the metric that was used in the computation.
            elapsed_time(float): the time that has passed since the evaluation
                computation started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print('Finished metric', kwargs['metric'].value, f'in {elapsed_time:1.4f}s')


def on_begin_eval_pipeline(event_listener, **kwargs):
    """Callback function when the evaluation pipeline started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_metrics(int): the number of metrics that will be processed.
    """

    if event_listener.verbose:
        print('\nStarting Evaluation Pipeline to process', kwargs['num_metrics'], 'metric(s)')


def on_end_eval_pipeline(event_listener, **kwargs):
    """Callback function when the evaluation pipeline started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_metrics(int): the number of metrics that were processed.
        elapsed_time(float): the time that has passed since the pipeline
            started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print(f'Finished Evaluation Pipeline on', kwargs['num_metrics'], 'metrics',
              f'in {elapsed_time:1.4f}s')


def on_begin_filter(event_listener, **kwargs):
    """Callback function when a model computation started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        filter_name(str): name of the filter.
    """
    if event_listener.verbose:
        print('Starting filter:', kwargs['filter_name'].value)


def on_end_filter(event_listener, **kwargs):
    """Callback function when a model computation finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        filter_name(str): the model that was used in the computation.
        elapsed_time(float): the time that has passed since the model
            computation started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print('Finished filter', kwargs['filter_name'], f'in {elapsed_time:1.4f}s')
