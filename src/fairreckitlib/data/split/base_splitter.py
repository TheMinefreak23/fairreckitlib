"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Any
import pandas as pd


class DataSplitter(metaclass=ABCMeta):
    """Base class for FairRecKit data splitters.

    A splitter is used to split a dataframe into a train and test set.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        """Construct the base splitter.

        Args:
            name: the name of the splitter.
            params: a dictionary containing the parameters for the splitter.
        """
        self.__name = name
        self.__params = params

    def get_name(self) -> str:
        """Get the name of the splitter.

        Returns:
            (str) the splitter name.
        """
        return self.__name

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the splitter.

        Returns:
            (dict) with the splitter parameters.
        """
        return dict(self.__params)

    @abstractmethod
    def run(self, dataframe: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the splitter on the specified dataframe.

        Args:
            dataframe: with at least the 'user' column.
            test_ratio: the fraction of users to use for testing.

        Returns:
            train_set, test_set: the train and test set of the split.
        """
        raise NotImplementedError()
