"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.factories import Factory, create_factory_from_list
from .filter_constants import KEY_DATA_FILTERS, FILTER_AGE, FILTER_GENDER, FILTER_COUNTRY
from .age import create_age_filter
from .gender import create_gender_filter
from .country import create_country_filter

def create_filter_factory() -> Factory:
    """Create a Factory with the following data filters:

    Age: a filter using a min and max value.
    Gender: a filter for female and male.
    Country: a filter for country names.

    Returns:
        The factory with all available filters.
    """
    return create_factory_from_list(KEY_DATA_FILTERS, [
        (FILTER_AGE,
         create_age_filter,
         None  # numerical params
         ),
        (FILTER_GENDER,
         create_gender_filter,
         None # catergorical params
         ),
        (FILTER_COUNTRY,
         create_country_filter, # count categorical params (country)
         None # count
         )
    ])
