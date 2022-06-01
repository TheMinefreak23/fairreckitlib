"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


import pandas as pd
from .base import DataFilter


class CountAttribFilter(DataFilter):
    """Filters out the dataframe on the count of a certain categorical column.
    input: count (int), dataframe with column country (or some other categorical column).
    body: Filter out all countries with less than count entities in the dataframe.
    output: dataframe
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        """The constructor.

        Args:
            data_frame: a pandas DataFrame being filtered
        """
        super().__init__(dataset)
        
    def run(self, col_name: str, filterLess_val: int = 0) -> pd.DataFrame:
        """Filter out the dataframe based on the column according to
        filterLess value.

        Args:
            col_name: the name of the column to be filtered
            filterLess_val: the amount to be filtered out on

        Returns:
            a filtered dataframe out of the given dataframe
        """
        if col_name in self.dataset.columns:
            df_agg = self.dataset[col_name].value_counts()
            mask = list(df_agg[df_agg.ge(filterLess_val)].index)
            df_filter = self.dataset[col_name].isin(mask)
            return self.dataset[df_filter].reset_index(drop=True)
        return self.dataset

    def __str__(self):
        """To string.

        Returns:
            name of the class
        """
        return self.__class__.__name__


def create_count_attrib_filter(data_frame: pd.DataFrame) -> DataFilter:
    """Create an instance of the class CountAttribFilter.

    Args:
        data_frame: a pandas DataFrame being filtered

    Returns:
        an instance of the CountAttribFilter class
    """
    return CountAttribFilter(data_frame)
