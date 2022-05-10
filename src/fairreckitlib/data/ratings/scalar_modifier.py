"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base_modifier import DataModifier


class ScalarModifier(DataModifier):
    """Scalar Modifier on data ratings.

    Applies a scalar to the rating column of the dataframe.
    """

    def run(self, dataframe, rating_type):
        # TODO apply scalar as specified in params
        raise NotImplementedError()
