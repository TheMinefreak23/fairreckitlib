"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..set.dataset import DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT
from .base_converter import RatingConverter


class RangeConverter(RatingConverter):
    """Range Converter on data ratings.

    Converts the rating column of the dataframe to a specified range.
    """

    def run(self, dataframe):
        """Converts ratings in the dataframe.

        Takes the max value and divides all values so
        they all fall within a range of [0,1], unless another upper
        bound is given by the rating_modifier. The rating can then
        also be multiplied by a scalar, eg when an implicit rating is needed.

        Args:
            dataframe(pandas.DataFrame): a df that should contain a 'rating' header.

        Returns:
            dataframe(pandas.DataFrame): with the converted rating values.
        """
        upper_bound = self.params['upper_bound']
        max_rating = dataframe.max()['rating']
        dataframe['rating'] = dataframe['rating'].apply(lambda x : x / max_rating * upper_bound)
        # any ratings above a 10 will be viewed as implicit
        return dataframe, DATASET_RATINGS_IMPLICIT if upper_bound > 10 else DATASET_RATINGS_EXPLICIT
