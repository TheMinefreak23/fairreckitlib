"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


from .constants import SPLIT_NAME, SPLIT_PARAMS

FUNC_CREATE_SPLIT = 'f_create_split'
FUNC_GET_SPLIT_PARAMS = 'f_get_split_params'

class SplitFactory:
    """Split Facotry with available split methods."""
    def __init__(self):
        self.__factory = {}

    def add(self, split_name, func_create_split, func_get_split_params):
        """Adds a split method to the factory.

        This function raises a KeyError when the name of the method
        is already registered for this split factory.

        Args:
            split_name(str): name of the split method
            func_create_split(function): creation function that takes the
                split parameters.
            func_get_split_params(function): get parameters function that
                describes all parameters for the split method.
        """
        if self.__factory.get(split_name) is not None:
            raise KeyError('Split method already exists: ' + split_name)

        self.__factory[split_name] = {
            FUNC_CREATE_SPLIT: func_create_split,
            FUNC_GET_SPLIT_PARAMS: func_get_split_params
        }

    def create(self, split_name, split_params):
        """Creates the split method with the specified name and parameters.
        
        This function raises a KeyError when the name of the split method
        is not registered for this split factory.
        
        Args:
            split_name(str): name of the split method.
            split_params(dict): parameters of the split method as name-value pairs.
        """
        if self.__factory.get(split_name) is None:
            raise KeyError('Split method does not exist: ' + split_name)

        func_create_split = self.__factory[split_name][FUNC_CREATE_SPLIT]
        split = func_create_split(split_params)
        split.name = split_name

        return split

    def get_split_params(self, split_name):
        """Gets the parameters for the specified split method.

        This function raises a KeyError when the name of the split method
        is not registered for this split factory.
        
        Args:
            split_name(str): name of the split method.
            
        Returns:
            split_params(SplitParameters): the parameters of the split method.
        """
        if split_name not in self.__factory:
            raise KeyError('Split method does not exist: ' + split_name)

        return self.__factory[split_name][FUNC_GET_SPLIT_PARAMS]()

    def get_available(self):
        """Gets the available split methods in the factory.
        
        Returns:
            split_list(array like): dict entries with splitter name and params.
        """
        split_list = []

        for split_name, entry in self.__factory.items():
            split_params = entry[FUNC_GET_SPLIT_PARAMS]()
            split_list.append({
                SPLIT_NAME: split_name,
                SPLIT_PARAMS: split_params
            })

        return split_list

    def get_available_split_names(self):
        """Gets the names of the available split methods in the factory.
        
        Returns:
            split_names(array like): list of split method names.
        """
        split_names = []

        for split_name, _ in self.__factory.items():
            split_names.append(split_name)

        return split_names

    def get_entries(self):
        """Gets the split method entries of the factory.
        
        Returns: 
            factory entries(dict)
        """
        return dict(self.__factory)

def create_split_factory_from_list(split_list):
    """Creates a SplitFactory from a list of tuples.
    
    Each tuple consists of:
    
    1) split method name
    2) split creation function
    3) split params funciton
    
    Args:
        split_list(array like): list of tuples.
    
    Returns:
        SplitFactory
    """
    factory = SplitFactory()

    for _, split in enumerate(split_list):
        (split_name, func_create_split, func_get_split_params) = split
        factory.add(
            split_name,
            func_create_split,
            func_get_split_params
        )

    return factory