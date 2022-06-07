"""This module contains a parser for the evaluation and metric configuration(s).

Functions:

    parse_evaluation_config: parse all metric configurations.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List

from ...core.config.config_factories import GroupFactory
from ...core.config.config_value_param import ConfigNumberParam
from ...core.core_constants import KEY_TOP_K
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_config_object import parse_config_object_list
from ..evaluation_factory import KEY_EVALUATION
from ..metrics.metric_constants import KEY_METRIC_PARAM_K
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
    # evaluation is not mandatory
    if not KEY_EVALUATION in experiment_config:
        return []

    eval_config = experiment_config[KEY_EVALUATION]

    top_k_param = {}
    if experiment_config.get(KEY_TOP_K, False):
        top_k_param[KEY_METRIC_PARAM_K] = ConfigNumberParam(
            KEY_METRIC_PARAM_K,
            int,
            experiment_config[KEY_TOP_K],
            (1, experiment_config[KEY_TOP_K])
        )

    parsed_config_objs = parse_config_object_list(
        KEY_EVALUATION,
        'metric',
        eval_config,
        metric_category_factory,
        event_dispatcher,
        params=top_k_param
    )

    metric_config_list = []
    for (metric, _) in parsed_config_objs:
        # TODO parse this
        metric_prefilters = []

        metric_config_list.append(MetricConfig(
            metric.name,
            metric.params,
            metric_prefilters
        ))

    return metric_config_list
