"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional, Tuple

from ...core.config_constants import KEY_NAME, KEY_PARAMS, KEY_TOP_K
from ...core.event_dispatcher import EventDispatcher
from ...core.factories import GroupFactory
from ...core.parsing.parse_assert import assert_is_type, assert_is_container_not_empty
from ...core.parsing.parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from ...core.parsing.parse_event import ON_PARSE
from ...core.parsing.parse_params import parse_config_parameters
from ..evaluation_factory import KEY_EVALUATION
from ..metrics.metric_factory import KEY_METRIC_PARAM_K, resolve_metric_factory
from .evaluation_config import MetricConfig


def parse_evaluation_config(
        experiment_config: Dict[str, Any],
        metric_category_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> List[MetricConfig]:
    """Parse all metric configurations.

    Args:
        experiment_config: the experiment's total configuration.
        metric_category_factory: the metric factory containing grouped available metrics.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        a list of parsed MetricConfig's which is possibly empty.
    """
    parsed_config = []

    # evaluation is not mandatory
    if not KEY_EVALUATION in experiment_config:
        return parsed_config

    eval_config = experiment_config[KEY_EVALUATION]

    # assert eval_config is a list
    if not assert_is_type(
        eval_config,
        list,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + KEY_EVALUATION + '\'',
        default=parsed_config
    ): return parsed_config

    # assert eval_config has list entries
    if not assert_is_container_not_empty(
        eval_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + KEY_EVALUATION + '\' is empty',
        default=parsed_config
    ): return parsed_config

    # parse eval_config list entries
    for _, metric_config in enumerate(eval_config):
        print(metric_config)
        metric, metric_name = parse_metric_config(
            metric_config,
            metric_category_factory,
            experiment_config.get(KEY_TOP_K),
            event_dispatcher
        )
        # skip on failure
        if metric is None:
            event_dispatcher.dispatch(
                ON_PARSE,
                msg='PARSE WARNING: failed to parse metric \'' +
                str(metric_name) + '\', skipping...'
            )
            continue

        parsed_config.append(metric)

    return parsed_config


def parse_metric_config(
        metric_config: Dict[str, Any],
        metric_category_factory: GroupFactory,
        top_k: int,
        event_dispatcher: EventDispatcher) -> Tuple[Optional[MetricConfig], Optional[str]]:
    """Parse a metric configuration.

    Args:
        metric_config: the metrics configuration.
        metric_category_factory: the metric factory containing grouped available metrics.
        top_k: the top_K value for recommendation and None for prediction.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration and the metric name or None on failure.
    """
    # assert dataset_config is a dict
    if not assert_is_type(
        metric_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid metric entry'
    ): return None, None

    # assert metric name is present
    if not assert_is_key_in_dict(
        KEY_NAME,
        metric_config,
        event_dispatcher,
        'PARSE ERROR: missing metric key \'' + KEY_NAME + '\' (required)'
    ): return None, None

    metric_name = metric_config[KEY_NAME]
    print(metric_name, metric_category_factory)
    metric_factory = resolve_metric_factory(metric_name, metric_category_factory)

    # assert metric name is available in the metric factory
    if not assert_is_one_of_list(
        metric_name,
        [] if metric_factory is None else metric_factory.get_available_names(),
        event_dispatcher,
        'PARSE ERROR: unknown metric name \'' + str(metric_name) + '\''
    ): return None, metric_name

    params = metric_factory.create_params(metric_name)

    top_k_param = params.get_param(KEY_METRIC_PARAM_K)
    # modify top_k param so that it will be parsed correctly
    if top_k and top_k_param:
        top_k_param.default_value = top_k
        top_k_param.max_value = top_k

    metric_params = params.get_defaults()

    # assert KEY_PARAMS is present
    # skip when the metric has no parameters at all
    if params.get_num_params() > 0 and assert_is_key_in_dict(
        KEY_PARAMS,
        metric_config,
        event_dispatcher,
        'PARSE WARNING: ' + metric_name + ' missing key \'' + KEY_PARAMS + '\'',
        default=metric_params
    ):
        # parse the metric parameters
        metric_params = parse_config_parameters(
            metric_config[KEY_PARAMS],
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
