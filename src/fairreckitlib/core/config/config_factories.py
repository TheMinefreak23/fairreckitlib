"""This module contains the (base) factories that are used in other packages.

Classes:

    BaseFactory: base class for all factories.
    Factory: class that instantiates new objects (a leaf).
    GroupFactory: class that groups other factories (a branch).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from ..core_constants import KEY_NAME, KEY_PARAMS
from .config_parameters import ConfigParameters, create_empty_parameters

FUNC_CREATE_OBJ = 'f_create_obj'
FUNC_CREATE_PARAMS = 'f_create_params'


class BaseFactory(metaclass=ABCMeta):
    """Base class for all FairRecKit experiment factories.

    The abstraction is intended to support functionality for a tree-like structure.
    The actual factory is a leaf and a group of factories that belong together a branch.

    Public methods:

    get_available
    get_available_names
    get_name
    is_available
    """

    def __init__(self, factory_name: str):
        """Construct the base factory.

        Args:
            factory_name: the name of the factory.
        """
        self.factory_name = factory_name
        self.factory = {}

    @abstractmethod
    def get_available(self) -> Any:
        """Get the available objects in the factory.

        The availability return type depends on the derived class.
        """
        raise NotImplementedError()

    def get_available_names(self) -> List[str]:
        """Get the names that are available in the factory.

        Returns:
            a list of names that is available.
        """
        result = []

        for name, _ in self.factory.items():
            result.append(name)

        return result

    def get_name(self) -> str:
        """Get the name of the factory.

        Returns:
            the factory name.
        """
        return self.factory_name

    def get_num_entries(self) -> int:
        """Get the number of entries present in the factory.

        Returns:
            the amount of entries.
        """
        return len(self.factory)

    @abstractmethod
    def is_obj_available(self, obj_name: str) -> bool:
        """Is the object with the specified name available.

        Args:
            obj_name: the name of the object to query for availability.

        Returns:
            whether the object is available.
        """
        raise NotImplementedError()


class Factory(BaseFactory):
    """The factory that implements object and parameters creation.

    Public methods:

    add_obj
    create
    create_params
    """

    def add_obj(self,
                obj_name: str,
                func_create_obj: Callable[[str, Dict[str, Any]], Any],
                func_create_obj_params: Callable[[], ConfigParameters]=None
                ) -> None:
        """Add object with associated parameter creation to the factory.

        Parameter creation is optional and should be None if the object has no parameters.
        A key error is raised when the obj_name is present in the factory.

        Args:
            obj_name: the name of the object
            func_create_obj: the function that creates and returns a new object.
            func_create_obj_params: the function that creates and returns the parameters
                that are associated with a newly created object.
        """
        if obj_name in self.factory:
            raise KeyError('Factory ' + self.factory_name,
                           ': object already exists: ' + obj_name)

        if func_create_obj_params is None:
            func_create_obj_params = create_empty_parameters

        self.factory[obj_name] = {
            FUNC_CREATE_OBJ: func_create_obj,
            FUNC_CREATE_PARAMS: func_create_obj_params
        }

    def create(self, obj_name: str, obj_params: Dict[str, Any]=None, **kwargs) -> Any:
        """Create and return a new object with the specified name.

        The specified parameters are expected to be of the same structure as the defaults
        of the ConfigParameters that are associated with the desired object. When no parameters
        are specified it will use the object's defaults.

        Args:
            obj_name: the name of the object to create.
            obj_params: the parameters of the object.

        Keyword Args:
            Any: extra arguments that need to be passed to the object on creation.

        Returns:
            the created object or None when it does not exist.
        """
        if obj_name not in self.factory:
            return None

        func_create_obj = self.factory[obj_name][FUNC_CREATE_OBJ]
        if obj_params is None:
            obj_params = self.create_params(obj_name).get_defaults()

        return func_create_obj(obj_name, dict(obj_params), **kwargs)

    def create_params(self, obj_name: str) -> ConfigParameters:
        """Create parameters for the object with the specified name.

        Args:
            obj_name: name of the object to create parameters for.

        Returns:
            the configuration parameters of the object or empty parameters when it does not exist.
        """
        if obj_name not in self.factory:
            return create_empty_parameters()

        return self.on_create_params(obj_name)

    def on_create_params(self, obj_name: str) -> ConfigParameters:
        """Create parameters for the object with the specified name.

        Args:
            obj_name: name of the object to create parameters for.

        Returns:
            the configuration parameters of the object or empty parameters when it does not exist.
        """
        return self.factory[obj_name][FUNC_CREATE_PARAMS]()

    def get_available(self) -> List[Dict[str, Dict[str, Any]]]:
        """Get the availability of all object names and their parameters.

        Each object in the factory has a name and parameters that consists of
        a dictionary with name-value pairs.

        Returns:
            a list of dictionary entries that includes the name and the parameters.
        """
        obj_list = []

        for obj_name, _ in self.factory.items():
            obj_list.append({
                KEY_NAME: obj_name,
                KEY_PARAMS: self.on_create_params(obj_name).to_dict()
            })

        return obj_list

    def is_obj_available(self, obj_name: str) -> bool:
        """Is the object with the specified name available.

        Checks the object for existing in this factory.

        Args:
            obj_name: the name of the object to query for availability.

        Returns:
            whether the object is available.
        """
        return obj_name in self.factory is not None


class GroupFactory(BaseFactory):
    """The factory that groups other factories together.

    Public methods:

    add_factory
    get_factory
    get_sub_availability
    """

    def add_factory(self, factory: BaseFactory) -> None:
        """Add the specified factory to the group.

        The name of the factory is used as the key, and thus it will raise a
        KeyError when the name of the factory already exists in the group.

        Args:
            factory: to add to the group.
        """
        factory_name = factory.get_name()
        if factory_name in self.factory:
            raise KeyError('Factory ' + self.factory_name,
                           ': factory already exists: ' + factory_name)

        self.factory[factory_name] = factory

    def get_available(self) -> Dict[str, Any]:
        """Get the availability of all factories in the group.

        Each factory has a name and availability that depends on the
        type of the factory. Effectively this will generate a tree-like
        structure of the factory's availability.

        Returns:
            a dictionary with factory name and availability pairs.
        """
        factory_list = {}

        for factory_name, factory in self.factory.items():
            factory_list[factory_name] = factory.get_available()

        return factory_list

    def get_factory(self, factory_name: str) -> BaseFactory:
        """Get the factory with the specified name.

        Args:
            factory_name: the name of the factory to retrieve

        Returns:
            the requested factory or None when not available.
        """
        return self.factory.get(factory_name)

    def get_sub_availability(
            self,
            sub_factory_name: str,
            sub_type: str = None) -> Dict[str, Any]:
        """Get the sub availability from the factory with the specified name (and type).

        Args:
            sub_factory_name: the name of the sub-factory to query for availability.
            sub_type: the subtype of the sub-factory to query for availability or None
                for the complete availability.

        Returns:
            a dictionary containing the availability of the sub-factory (type).
        """
        sub_factory = self.get_factory(sub_factory_name)
        if sub_type is None:
            return sub_factory.get_available()

        type_factory = sub_factory.get_factory(sub_type)
        if type_factory is None:
            return {}

        return type_factory.get_available()

    def is_obj_available(self, obj_name: str) -> bool:
        """Is the object with the specified name available.

        Checks the object for existing in any of the child factories.

        Args:
            obj_name: the name of the object to query for availability.

        Returns:
            whether the object is available.
        """
        for _, factory in self.factory.items():
            if factory.is_obj_available(obj_name):
                return True

        return False


def create_factory_from_list(
        factory_name: str,
        obj_tuple_list: List[Tuple[
            str,
            Callable[[str, Dict[str, Any]], Any],
            Callable[[], ConfigParameters]
        ]]) -> Factory:
    """Create and return the factory with the specified tuple entries.

    Each tuple in the list consists of three things; the name of the object,
    the object creation function and the parameter creation function.

    Args:
        factory_name: the name of the factory to create.
        obj_tuple_list: a list of object tuples to add after factory creation.

    Returns:
        the factory with the added objects.
    """
    factory = Factory(factory_name)

    for _, obj in enumerate(obj_tuple_list):
        (obj_name, func_create_obj, func_create_obj_params) = obj
        factory.add_obj(
            obj_name,
            func_create_obj,
            func_create_obj_params
        )

    return factory
