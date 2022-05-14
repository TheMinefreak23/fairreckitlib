"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import pandas as pd


class DataModifier(metaclass=ABCMeta):
    """Base class for FairRecKit data modifiers.

    Public methods:

    get_name
    get_params
    run
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        """Construct the DataModifier.

        Args:
            name: the name of the modifier.
            params: the modifier parameters.
        """
        self.name = name
        self.params = params

    def get_name(self) -> str:
        """Get the name of the modifier.

        Returns:
            the modifier name.
        """
        return self.name

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the modifier.

        Returns:
            the modifier parameters.
        """
        return dict(self.params)

    @abstractmethod
    def run(self, dataframe: pd.DataFrame) -> Any:
        """Run the modifier on the specified dataframe.

        Args:
            dataframe: source df to modify.

        Returns:
            any modification to the dataframe.
        """
        raise NotImplementedError()
