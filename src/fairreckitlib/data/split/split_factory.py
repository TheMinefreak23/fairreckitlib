"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict
import time

from ...core.config_params import ConfigParameters
from ...core.factories import Factory, create_factory_from_list
from .random_splitter import RandomSplitter
from .temporal_splitter import TemporalSplitter

SPLIT_RANDOM = 'random'
SPLIT_TEMPORAL = 'temporal'

KEY_SPLITTING = 'splitting'
KEY_SPLIT_TEST_RATIO = 'test_ratio'

DEFAULT_SPLIT_TEST_RATIO = 0.2
DEFAULT_SPLIT_TYPE = SPLIT_RANDOM


def create_split_factory() -> Factory:
    """Create a Factory with all data splitters.

    Returns:
        (Factory) with all splitters.
    """
    return create_factory_from_list(KEY_SPLITTING, [
        (SPLIT_RANDOM,
         _create_random_splitter,
         _create_random_splitter_params
         ),
        (SPLIT_TEMPORAL,
         _create_temporal_splitter,
         None
         )
    ])


def _create_random_splitter(name: str, params: Dict[str, Any]) -> RandomSplitter:
    """Create the Random Splitter.

    Returns:
        (RandomSplitter) the random data splitter.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return RandomSplitter(name, params)


def _create_random_splitter_params() -> ConfigParameters:
    """Get the parameters of the random splitter.

    Returns:
        params(ConfigParameters) the params of the splitter.
    """
    params = ConfigParameters()
    params.add_random_seed('seed')
    return params

def _create_temporal_splitter(name: str, params: Dict[str, Any]) -> TemporalSplitter:
    """Create the Temporal Splitter.

    Returns:
        (TemporalSplitter) the temporal data splitter.
    """
    return TemporalSplitter(name, params)
