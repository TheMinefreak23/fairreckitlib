"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .gender import GenderFilter, create_gender_filter

FILTER_GENDER = 'gender'

def get_filter():
    return {
        FILTER_GENDER: create_gender_filter
    }