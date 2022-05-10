"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.config_params import ConfigParameters
from ...core.factories import GroupFactory, create_factory_from_list
from ..set.dataset import DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT
from .range_modifier import RangeModifier
from .scalar_modifier import ScalarModifier

KEY_RATING_MODIFIER = 'rating_modifier'

MODIFIER_RANGE = 'range'
MODIFIER_SCALAR = 'scalar'


def create_rating_modifier_factory():
    rating_modifier_factory = GroupFactory(KEY_RATING_MODIFIER)
    rating_modifier_factory.add_factory(create_explicit_rating_modifier_factory())
    rating_modifier_factory.add_factory(create_implicit_rating_modifier_factory())
    return rating_modifier_factory


def create_explicit_rating_modifier_factory():
    return create_factory_from_list(DATASET_RATINGS_EXPLICIT, [
        (MODIFIER_SCALAR,
         _create_scalar_modifier,
         _create_scalar_modifier_params
         )
    ])


def create_implicit_rating_modifier_factory():
    return create_factory_from_list(DATASET_RATINGS_IMPLICIT, [
        (MODIFIER_RANGE,
         _create_range_modifier,
         _create_range_modifier_params
         )
    ])


def _create_range_modifier(name, params):
    return RangeModifier(name, params)


def _create_range_modifier_params():
    methods = [
        'none',
        'APC',
        'ALC'
    ]

    params = ConfigParameters()
    params.add_value('upper_bound', float, 1.0, (1.0, 1000.0))
    params.add_option('method', str, methods[0], methods)
    return params


def _create_scalar_modifier(name, params):
    return ScalarModifier(name, params)


def _create_scalar_modifier_params():
    params = ConfigParameters()
    params.add_value('value', float, 10.0, (1.00, 1000.0))
    return params
