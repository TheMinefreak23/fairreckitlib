"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base_modifier import DataModifier


class RangeModifier(DataModifier):
    """Range Modifier on data ratings.

    Restricts the rating column of the dataframe to a specified range
    and applies an (optional) method beforehand.
    """

    def run(self, dataframe, rating_type):
        # TODO apply method as specified in params and restrict to range
        raise NotImplementedError()
