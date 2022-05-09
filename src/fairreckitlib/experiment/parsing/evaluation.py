"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.events import config_event
from src.fairreckitlib.experiment.parsing import assertion
from ..config import MetricConfig
from ..constants import EXP_KEY_EVALUATION
from ..constants import EXP_KEY_METRIC_PARAM_K
from ..constants import EXP_KEY_OBJ_NAME
from ..constants import EXP_KEY_OBJ_PARAMS
from ..constants import EXP_KEY_TOP_K
from .params import parse_config_parameters


def parse_evaluation_config(experiment_config, metric_factory, event_dispatcher):
    """Parses all metric configurations.

    Args:
        experiment_config(dict): the experiment's total configuration.
        metric_factory(MetricFactory): the metric factory containing available metrics.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(array like): list of parsed MetricConfig's.
    """
    parsed_config = []

    # evaluation is not mandatory
    if not EXP_KEY_EVALUATION in experiment_config:
        return parsed_config

    eval_config = experiment_config[EXP_KEY_EVALUATION]

    # assert eval_config is a list
    if not assertion.is_type(
        eval_config,
        list,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + EXP_KEY_EVALUATION + '\'',
        default=parsed_config
    ): return parsed_config

    # assert eval_config has list entries
    if not assertion.is_container_not_empty(
        eval_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + EXP_KEY_EVALUATION + '\' is empty',
        default=parsed_config
    ): return parsed_config

    # parse eval_config list entries
    for _, metric_config in enumerate(eval_config):
        metric, metric_name = parse_metric_config(
            metric_config,
            metric_factory,
            experiment_config.get(EXP_KEY_TOP_K),
            event_dispatcher
        )
        # skip on failure
        if metric is None:
            event_dispatcher.dispatch(
                config_event.ON_PARSE,
                msg='PARSE WARNING: failed to parse metric \'' +
                str(metric_name) + '\', skipping...'
            )
            continue

        parsed_config.append(metric)

    return parsed_config

def parse_metric_config(metric_config, metric_factory, top_k, event_dispatcher):
    """Parses a metric configuration.

    Args:
        metric_config(dict): the metrics configuration.
        metric_factory(MetricFactory): the metric factory containing available metrics.
        top_k(int): the top_K value for recommendation and None for prediction.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(MetricConfig): the parsed configuration or None on failure.
        metric_name(str): the name of the parsed metric or None on failure.
    """
    # assert dataset_config is a dict
    if not assertion.is_type(
        metric_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid metric entry'
    ): return None, None

    # assert metric name is present
    if not assertion.is_key_in_dict(
        EXP_KEY_OBJ_NAME,
        metric_config,
        event_dispatcher,
        'PARSE ERROR: missing metric key \'' + EXP_KEY_OBJ_NAME + '\' (required)'
    ): return None, None

    metric_name = metric_config[EXP_KEY_OBJ_NAME]

    if top_k is None:
        available_metrics = metric_factory.get_available_prediction_metric_names()
    else:
        available_metrics = metric_factory.get_available_recommendation_metric_names()

    # assert metric name is available in the metric factory
    if not assertion.is_one_of_list(
        metric_name,
        available_metrics,
        event_dispatcher,
        'PARSE ERROR: unknown metric name \'' + str(metric_name) + '\''
    ): return None, metric_name

    params = metric_factory.get_params(metric_name)

    top_k_param = params.get_param(EXP_KEY_METRIC_PARAM_K)
    # modify top_k param so that it will be parsed correctly
    if top_k and top_k_param:
        top_k_param.default_value = top_k
        top_k_param.max_value = top_k

    metric_params = params.get_defaults()

    # assert EXP_KEY_METRIC_PARAMS is present
    # skip when the metric has no parameters at all
    if params.get_num_params() > 0 and assertion.is_key_in_dict(
        EXP_KEY_OBJ_PARAMS,
        metric_config,
        event_dispatcher,
        'PARSE WARNING: ' + metric_name + ' missing key \'' + EXP_KEY_OBJ_PARAMS + '\'',
        default=metric_params
    ):
        # parse the metric parameters
        metric_params = parse_config_parameters(
            metric_config[EXP_KEY_OBJ_PARAMS],
            metric_name,
            params,
            event_dispatcher
        )

    # TODO parse this
    metric_prefilters = []

    parsed_config = MetricConfig(
        metric_name,
        metric_params,
        metric_prefilters
    )

    return parsed_config, metric_name
