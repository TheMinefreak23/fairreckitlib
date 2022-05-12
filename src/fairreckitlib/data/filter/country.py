"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base import DataFilter


class CountryFilter(DataFilter):
    """Filters the dataframe on country, if such column exists."""
    def __init__(self, df):
        self.df = df
    
    def run(self, df, country):
        """Country == name of the country that you want to filter on. 
        All other countries get filtered out.
        """
        if 'country' in df.columns:
            filter = df['country'] == country
            return df[filter]
        else: return df

def create_country_filter(df):
    return CountryFilter(df)