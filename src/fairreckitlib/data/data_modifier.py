"""This module contains the base class and factory for data modification.

Classes:

    DataModifier: the base class for data modifying.
    DataModifierFactory: the factory that creates data modifiers related to a dataset matrix.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd

from ..core.config.config_factories import Factory, GroupFactory, FUNC_CREATE_PARAMS
from ..core.config.config_parameters import ConfigParameters
from .set.dataset import Dataset
from .set.dataset_registry import DataRegistry

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


class DataModifierFactory(Factory):
    """Factory for data modifier creation.

    The intended use is to associate the factory with a specific matrix of a dataset.
    Both the created parameters and the created data modifiers are supplied
    with a reference to the dataset and the name of the matrix they belong to.
    """

    def __init__(self, matrix_name: str, dataset: Dataset):
        """Construct the DataModifierFactory.

        Args:
            matrix_name: the name of the matrix that it relates to.
            dataset: the dataset associated with the matrix.
        """
        Factory.__init__(self, matrix_name)
        self.dataset = dataset

    def create(self, obj_name: str, obj_params: Dict[str, Any]=None, **kwargs) -> DataModifier:
        """Create and return a new data modifier with the specified name.

        The specified parameters are expected to be of the same structure as the defaults
        of the ConfigParameters that are associated with the desired data modifier.
        When no parameters are specified it will use the data modifier's defaults.

        Args:
            obj_name: the name of the data modifier to create.
            obj_params: the parameters of the data modifier.

        Keyword Args:
            Any: extra arguments that need to be passed to the data modifier on creation.

        Returns:
            the created data modifier or None when it does not exist.
        """
        kwargs['dataset'] = self.dataset
        kwargs['matrix_name'] = self.factory_name
        return Factory.create(self, obj_name, obj_params, **kwargs)

    def on_create_params(self, obj_name: str) -> ConfigParameters:
        """Create parameters for the data modifier with the specified name.

        Args:
            obj_name: name of the data modifier to create parameters for.

        Returns:
            the configuration parameters of the object or empty parameters when it does not exist.
        """
        kwargs = {
            'column_name': obj_name,
            'dataset': self.dataset,
            'matrix_name': self.factory_name
        }
        return self.factory[obj_name][FUNC_CREATE_PARAMS](**kwargs)


def create_data_modifier_factory(
        data_registry: DataRegistry,
        factory_name: str,
        func_on_add_entries: Callable[[DataModifierFactory, Dataset], None]) -> GroupFactory:
    """Create a data modifier factory for each dataset-matrix pair.

    Args:

        data_registry: the data registry with available datasets.
        factory_name: the name of the data modifier factory.
        func_on_add_entries: callback for each dataset-matrix pair to add data modifiers.

    Returns:
        the factory with all available data modifiers per dataset-matrix pair.
    """
    factory = GroupFactory(factory_name)

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)
        dataset_factory = GroupFactory(dataset.get_name())

        factory.add_factory(dataset_factory)

        for matrix_name in dataset.get_available_matrices():
            matrix_factory = DataModifierFactory(matrix_name, dataset)
            func_on_add_entries(matrix_factory, dataset)
            dataset_factory.add_factory(matrix_factory)

    return factory
