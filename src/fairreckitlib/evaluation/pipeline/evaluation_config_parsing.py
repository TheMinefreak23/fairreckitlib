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
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs
from ...data.filter.filter_config_parsing import parse_data_subset_config
from ...data.set.dataset_registry import DataRegistry
from ..evaluation_factory import KEY_EVALUATION
from ..metrics.metric_constants import KEY_METRIC_PARAM_K, KEY_METRIC_SUBGROUP
from .evaluation_config import MetricConfig


def parse_evaluation_config(
        data_registry: DataRegistry,
        data_filter_factory: GroupFactory,
        experiment_config: Dict[str, Any],
        metric_category_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> List[MetricConfig]:
    """Parse all metric configurations.

    Args:
        data_registry: the data registry with available dataset matrices.
        data_filter_factory: the dataset filter group factory.
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
    # override top K param according to experiment configuration
    if experiment_config.get(KEY_TOP_K, False):
        top_k_param[KEY_METRIC_PARAM_K] = ConfigNumberParam(
            KEY_METRIC_PARAM_K,
            int,
            experiment_config[KEY_TOP_K],
            (1, experiment_config[KEY_TOP_K])
        )

    # parse metric configurations as objects
    parsed_config_objs = parse_config_object_list(
        KEY_EVALUATION,
        'metric',
        eval_config,
        metric_category_factory,
        event_dispatcher,
        params=top_k_param
    )

    # convert object to metric configurations
    metric_config_list = []
    for (metric, metric_config) in parsed_config_objs:
        metric_subgroup = None

        # attempt to parse metric subgroup
        if KEY_METRIC_SUBGROUP in metric_config:
            metric_data_subset, _ = parse_data_subset_config(
                metric_config[KEY_METRIC_SUBGROUP],
                data_registry,
                data_filter_factory,
                event_dispatcher,
                data_parent_name='metric \'' + metric.name + '\'',
                required=False
            )

            # skip on failure or no filter passes
            if metric_data_subset is not None and len(metric_data_subset.filter_passes) == 0:
                event_dispatcher.dispatch(ParseEventArgs(
                    ON_PARSE,
                    'PARSE WARNING: metric \'' + metric.name +
                    '\' data filter passes are empty, skipping...'
                ))

            metric_subgroup = metric_data_subset

        metric_config_list.append(MetricConfig(
            metric.name,
            metric.params,
            metric_subgroup
        ))

    return metric_config_list
