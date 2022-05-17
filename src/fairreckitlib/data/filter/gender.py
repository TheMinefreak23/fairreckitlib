"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import pandas as pd
from .base import DataFilter


class GenderFilter(DataFilter):
    """Filters the dataframe on gender column f/m, if such a column exists."""
    def run(self, gender: str = None) -> pd.DataFrame:
        """
        Filters specific gender of the dataframe.

        Args:
            country: the name of the country used in filtering

        Returns:
            a filtered dataframe from the given dataframe
        """
        if 'gender' in self.dataset.columns:
            df_filter = []
            gender = gender.lower()
            if gender in ['female', 'f']:
                df_filter = (self.dataset.gender.lower().eq('f') |
                             self.dataset.gender.lower().eq('female'))
            elif gender in ['male', 'm']:
                df_filter = (self.dataset.gender.lower().eq('male') |
                             self.dataset.gender.lower().eq('m'))
            else: return self.dataset
            return self.dataset[df_filter]
        return self.dataset

    def __str__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__class__.__name__

def create_gender_filter(data_frame: pd.DataFrame) -> DataFilter:
    """
    Creates an instance of the class GenderFilter

    Args:
        data_frame: a pandas DataFrame being filtered

    Returns:
        an instance of the GenderFilter class
    """
    return GenderFilter(data_frame)
