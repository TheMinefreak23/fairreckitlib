
"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from typing import Any, Dict
import pandas as pd
from .base import DataFilter


class GenderFilter(DataFilter):
    """Filters the dataframe on gender column f/m, if such a column exists."""

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Filter specific gender of the dataframe.

        Args:
            dataframe: the dataframe this filter needs to be applied on.

        Returns:
            a filtered dataframe from the given dataframe
        """
        if 'gender' in dataframe.columns:
            df_filter = []
            gender = self.params['gender'].lower()
            if gender in ['female', 'f']:
                df_filter = (dataframe.gender.map(lambda x: x.lower() if x else x).eq('f') |
                             dataframe.gender.map(lambda x: x.lower() if x else x).eq('female'))
            elif gender in ['male', 'm']:
                df_filter = (dataframe.gender.map(lambda x: x.lower() if x else x).eq('male') |
                             dataframe.gender.map(lambda x: x.lower() if x else x).eq('m'))
            else: return dataframe
            return dataframe[df_filter].reset_index(drop=True)
        return dataframe

    def __str__(self):
        """To string

        Returns:
            name of the class
        """
        return self.__class__.__name__

def create_gender_filter(name: str, 
                         params: Dict[str, Any], 
                         **kwargs) -> DataFilter:
    """Create an instance of the class GenderFilter

    Args:
        name: UserGender
        params: Dictionary with only one key: 'gender'
        **kwargs (Optional): Not used.

    Returns:
        an instance of the GenderFilter class
    """
    return GenderFilter(name, params)
