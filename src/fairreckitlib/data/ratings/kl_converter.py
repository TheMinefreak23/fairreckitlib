"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
from .base_converter import RatingConverter


class KLConverter(RatingConverter):
    """Kullback-Leibler Converter on data ratings.

    Applies the Kullback-Leibler formula to the rating column of the dataframe.
    """

    def run(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Apply the Kullback-Leibler formula to convert ratings.

        Args:
            dataframe: with user, item, rating columns.

        Returns:
            dataframe with converted ratings.
            (str) rating type, which is now explicit.

        """
        method = self.params['method']
        # TODO apply kullback-leibler formula on the rating column. Needs APC/ALC arg.
        raise NotImplementedError()
