"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .age import create_age_filter
from .country import create_country_filter
from .gender import create_gender_filter

FILTER_AGE = 'age'
FILTER_COUNTRY = 'country'
FILTER_GENDER = 'gender'

def get_filter():
    """_summary_

    Returns:
        _type_: _description_
    """
    return {
        FILTER_AGE: create_age_filter,
        FILTER_COUNTRY: create_country_filter,
        FILTER_GENDER: create_gender_filter
    }
