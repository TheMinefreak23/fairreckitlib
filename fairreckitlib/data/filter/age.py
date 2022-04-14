"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas
from .base import DataFilter

class AgeFilter(DataFilter):
    """Filters the dataframe on user age, if such a column exists."""
    def __init__(self, df):
            self.df = df
    
    def run(self, df, min=0, max=100): 
        if 'age' in df.columns:
            filter = df['age'] >= min and df['age'] <= max
            return df.loc[filter]
        else: return df

def create_age_filter():
    return AgeFilter()
