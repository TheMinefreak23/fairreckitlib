"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.experiment.config import EXP_KEY_DATASET_RATING_MODIFIER
from src.fairreckitlib.core.factory import create_factory_from_list
from src.fairreckitlib.core.params import ConfigParameters

SPLIT_RANDOM = 'random'
SPLIT_TEMPORAL = 'temporal'


def create_rating_modifier_factory():
    """Creates a Factory with all data modifiers.

    Returns:
        (Factory) with all rating modifiers.
    """
    return create_factory_from_list(EXP_KEY_DATASET_RATING_MODIFIER, [
        (SPLIT_RANDOM,
         _create_random_splitter,
         _create_random_splitter_params
         ),
        (SPLIT_TEMPORAL,
         _create_temporal_splitter,
         None
         )
    ])


def _create_random_splitter(name, params):
    """Creates the Random Splitter.

    Returns:
        (RandomSplitter) the random data splitter.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return RandomSplitter(name, params)


def _create_random_splitter_params():
    """Gets the parameters of the random splitter.

    Returns:
        params(ConfigParameters) the params of the splitter.
    """
    params = ConfigParameters()
    params.add_random_seed('seed')
    return params

def _create_temporal_splitter(name, params):
    """Creates the Temporal Splitter.

    Returns:
        (TemporalSplitter) the temporal data splitter.
    """
    return TemporalSplitter(name, params)
