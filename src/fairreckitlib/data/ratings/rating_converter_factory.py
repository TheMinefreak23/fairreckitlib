"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.config_params import ConfigParameters
from ...core.factories import create_factory_from_list
from .range_converter import RangeConverter
from .kl_converter import KLConverter

KEY_RATING_CONVERTER = 'rating_converter'

CONVERTER_RANGE = 'range'
CONVERTER_KL = 'kl'


def create_rating_converter_factory():
    return create_factory_from_list(KEY_RATING_CONVERTER, [
        (CONVERTER_KL,
         _create_kl_converter,
         _create_kl_converter_params
         ),
        (CONVERTER_RANGE,
         _create_range_converter,
         _create_range_converter_params
         )
    ])


def _create_range_converter(name, params):
    return RangeConverter(name, params)


def _create_range_converter_params():
    params = ConfigParameters()
    params.add_value('upper_bound', float, 1.0, (1.0, 1000.0))
    return params


def _create_kl_converter(name, params, **kwargs):
    return KLConverter(name, params)


def _create_kl_converter_params():
    methods = [
        'APC',
        'ALC'
    ]

    params = ConfigParameters()
    params.add_option('method', str, methods[0], methods)
    return params
