"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

ON_BEGIN_EXP = 'on_begin_experiment'
ON_END_EXP = 'on_end_experiment'
ON_BEGIN_THREAD_EXP = 'on_begin_thread_experiment'
ON_END_THREAD_EXP = 'on_end_thread_experiment'


def get_experiment_events():
    """Gets all experiment pipeline events.

    Returns:
        (array like) list of pairs in the format (event_id, func_on_event)
    """
    return [
        (ON_BEGIN_EXP, on_begin_experiment),
        (ON_END_EXP, on_end_experiment),
        (ON_BEGIN_THREAD_EXP, on_begin_thread_experiment),
        (ON_END_THREAD_EXP, on_end_thread_experiment)
    ]


def on_begin_experiment(event_listener, **kwargs):
    """Callback function when a model computation started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        experiment_name(str): name of the experiment (run).
    """
    if event_listener.verbose:
        print('Starting experiment', kwargs['experiment_name'])


def on_end_experiment(event_listener, **kwargs):
    """Callback function when a model computation finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        experiment_name(str): name of the experiment (run).
        elapsed_time(float): the time that has passed since the model
            computation started, expressed in seconds.
    """
    if event_listener.verbose:
            elapsed_time = kwargs['elapsed_time']
            print('Finished experiment', kwargs['experiment_name'], f'in {elapsed_time:1.4f}s')


def on_begin_thread_experiment(event_listener, **kwargs):
    """Callback function when a model computation started.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_runs(int): number of experiments to run.
        experiment_name(str): name of the experiment (run).
    """
    if event_listener.verbose:
        print('Starting', kwargs['num_runs'], 'experiment(s) with name', kwargs['experiment_name'])


def on_end_thread_experiment(event_listener, **kwargs):
    """Callback function when a model computation finished.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_runs(int): number of experiments that ran.
        experiment_name(str): name of the experiment (run).
        elapsed_time(float): the time that has passed since the model
            computation started, expressed in seconds.
    """
    if event_listener.verbose:
        print('Finished', kwargs['num_runs'], 'experiment(s) with name', kwargs['experiment_name'])



