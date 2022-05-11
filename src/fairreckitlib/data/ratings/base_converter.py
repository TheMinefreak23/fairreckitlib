"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class RatingConverter(metaclass=ABCMeta):
    """Base class for FairRecKit rating converters.

    A converter is used to convert ratings of a dataframe.
    """
    def __init__(self, name, params):
        self.__name = name
        self.__params = params

    def get_name(self):
        """Gets the name of the converter.

        Returns:
            (str) the converter name.
        """
        return self.__name

    def get_params(self):
        """Get the parameters of the converter.

        Returns:
            (dict) with the converter parameters.
        """
        return dict(self.__params)

    @abstractmethod
    def run(self, dataframe):
        """Runs the converter on the specified dataframe.

        Args:
            dataframe(pandas.DataFrame): with at least the 'rating' column.

        Returns:
            dataframe(pandas.DataFrame): the converter dataframe.
            rating_type(str): the converted type of rating, either 'explicit' or 'implicit'.
        """
        raise NotImplementedError()
