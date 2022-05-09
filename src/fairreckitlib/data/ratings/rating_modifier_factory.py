"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.factory import create_factory_from_list
from .range_modifier import RangeModifier
from .scalar_modifier import ScalarModifier

RATING_MODIFIER_KEY = 'rating_modifier'

MODIFIER_RANGE = 'range'
MODIFIER_SCALAR = 'scalar'


def create_rating_modifier_factory():
    """Creates a Factory with all data modifiers.

    Returns:
        (Factory) with all rating modifiers.
    """
    return create_factory_from_list(RATING_MODIFIER_KEY, [
        (MODIFIER_RANGE,
         _create_range_modifier,
         None
         ),
        (MODIFIER_SCALAR,
         _create_scalar_modifier,
         None
         )
    ])


def _create_range_modifier(name, params):
    """Creates the Range Modifier.

    Returns:
        (RangeModifier) the range rating modifier.
    """
    return RangeModifier(name, params)


def _create_scalar_modifier(name, params):
    """Creates the Scalar Modifier.

    Returns:
        (ScalarModifier) the scalar rating modifier.
    """
    return ScalarModifier(name, params)
