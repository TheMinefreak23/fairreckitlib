"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base import DataFilter


class GenderFilter(DataFilter):
    """Filters the dataframe on gender column f/m, if such a column exists."""
    def __init__(self, df):
        self.df = df
    
    def run(self, df, gender):
        """Gender == 'female' to keep only female users. Otherwise only male users."""
        if 'gender' in df.columns:
            if gender == 'female': filter = df['gender'] == 'f' or df['gender'] == 'female'
            else: filter = df['gender'] == 'm' or df['gender'] == 'male'
            return df.loc[filter]
        else: return df

def create_gender_filter(df):
    return GenderFilter(df)