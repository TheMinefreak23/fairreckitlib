"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas
from .base import DataFilter

class CountryFilter(DataFilter):
    """Filters the dataframe on country, if such a column exists."""
    def __init__(self, df):
        self.df = df
    
    def run(self, df, filter_value):
        if 'country' in df.columns:
            filter = df['country'] == filter_value
            return df.loc[filter]
        else: return df

def create_country_filter():
    return CountryFilter()