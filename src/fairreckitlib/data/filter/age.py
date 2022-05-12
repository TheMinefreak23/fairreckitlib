"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base import DataFilter


class AgeFilter(DataFilter):
    """Filters the dataframe on user age, if such a column exists."""
    def __init__(self, df):
            self.df = df
    
    def run(self, min=0, max=100): 
        """Min and max allowed age of the users. Default is 0 & 100."""
        if 'age' in self.df.columns:
            # filter = self.df['age'].ge(min) & self.df['age'].le(max)
            filter = self.df['age'].between(min, max, inclusive="both")
            return self.df[filter]
        else: return self.df

def create_age_filter(df):
    return AgeFilter(df)
