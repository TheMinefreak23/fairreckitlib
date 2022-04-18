"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

ON_BEGIN_EVAL_PIPELINE = 'on_begin_eval_pipeline'
ON_END_EVAL_PIPELINE = 'on_end_eval_pipeline'
ON_BEGIN_EVAL = 'on_begin_eval'
ON_END_EVAL = 'on_end_eval'
ON_BEGIN_FILTER = 'on_begin_filter'
ON_END_FILTER = 'on_end_filter'


def get_evaluation_events():
    """TODO"""
    # TODO
    return [(ON_BEGIN_EVAL_PIPELINE, on_begin_eval_pipeline)
            (ON_END_EVAL_PIPELINE, on_end_eval_pipeline)
            (ON_END_EVAL, on_end_eval)
            (ON_BEGIN_FILTER, on_begin_filter)
            (ON_END_FILTER, on_end_filter)
            ]

def on_begin_eval(event_listener, **kwargs):
    """Callback function when a model computation started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        model_name(str): name of the model (algorithm).
    """
    if event_listener.verbose:
        print('Starting model:', kwargs['model_name'])


def on_end_eval(event_listener, **kwargs):
    """Callback function when a model computation finished.

        Args:
            event_listener(object): the listener that is registered
                in the event dispatcher with this callback.

        Keyword Args:
            metric(Metric): the metric that was used in the computation.
            elapsed_time(float): the time that has passed since the model
                computation started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        print('Finished model:', kwargs['model'].name, f'in {elapsed_time:1.4f}s')


def on_begin_eval_pipeline(event_listener, **kwargs):
    """Callback function when the evaluation pipeline started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_metrics(int): the number of metrics that will be processed.
    """

    if event_listener.verbose:
        print('\nStarting Model Pipeline:', kwargs['api_name'],
              'to process', kwargs['num_models'], 'model(s)')


def on_end_eval_pipeline(event_listener, **kwargs):
    """Callback function when the evaluation pipeline ended.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_models(int): the number of models that were processed.
        elapsed_time(float): the time that has passed since the pipeline
            started, expressed in seconds.
    """

    if event_listener.verbose:
        print('\nStarting Model Pipeline:', kwargs['api_name'],
              'to process', kwargs['num_models'], 'model(s)')



def on_begin_filter():


def on_end_filter():