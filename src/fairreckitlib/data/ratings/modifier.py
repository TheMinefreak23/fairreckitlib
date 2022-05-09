"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class DataModifier(metaclass=ABCMeta):

    def __init__(self, name, params):
        self.__name = name
        self.__params = params

    def get_name(self):
        """Gets the name of the modifier.

        Returns:
            (str) the modifier name.
        """
        return self.__name

    def get_params(self):
        """Get the parameters of the modifier.

        Returns:
            (dict) with the modifier parameters.
        """
        return dict(self.__params)

    @abstractmethod
    def run(self, dataframe, rating_type):
        """Runs the splitter on the specified dataframe.

        Args:
            dataframe(pandas.DataFrame): with at least the 'rating' column.
            rating_type(str): the type of rating, either 'explicit' or 'implicit'.

        Returns:
            dataframe(pandas.DataFrame): the modified dataframe.
            rating_type(str): the modified type of rating, either 'explicit' or 'implicit'.
        """
        raise NotImplementedError()
