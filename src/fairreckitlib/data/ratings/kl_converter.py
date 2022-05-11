"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base_converter import RatingConverter


class KLConverter(RatingConverter):
    """Kullback-Leibler Converter on data ratings.

    Applies the Kullback-Leibler formula to the rating column of the dataframe.
    """

    def run(self, dataframe):
        # TODO apply kullback-leibler formula on the rating column. Needs APC/ALC arg.
        raise NotImplementedError()
