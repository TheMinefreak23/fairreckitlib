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
    
    def run(self, gender):
        """Gender == 'female' to keep only female users. Otherwise only male users."""
        if 'gender' in self.df.columns:
            filter = []
            gender = gender.lower()
            if gender in ['female', 'f']:
                filter = self.df.gender.lower().eq('f') | self.df.gender.lower().eq('female')
            elif gender in ['male', 'm']:
                filter = self.df.gender.lower().eq('male') | self.df.gender.lower().eq('m')
            else: return self.df
            return self.df[filter]
        else: return self.df

def create_gender_filter(df):
    return GenderFilter(df)