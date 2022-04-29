"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .constants import ALGORITHM_NAME
from .constants import ALGORITHM_PARAMS

FUNC_GET_ALGORITHM_PARAMS = 'f_get_algorithm_params'
FUNC_CREATE_ALGORITHM = 'f_create_algorithm'


class AlgorithmFactory:
    """Algorithm Factory with available algorithms.

    Args:
        api_name(str): name of the API associated with the algorithm factory.
    """
    def __init__(self, api_name):
        self.__api_name = api_name
        self.__factory = {}

    def add(self, algo_name, func_create_algo, func_get_algo_params):
        """Adds an algorithm to the factory.

        This function raises a KeyError when the name of the algorithm
        is already registered for this algorithm factory.

        Args:
            algo_name(str): name of the algorithm.
            func_create_algo(function): creation function that takes the
                algorithm parameters and **kwargs.
            func_get_algo_params(function): get parameters function that
                describes all parameters for the algorithm.
        """
        if self.__factory.get(algo_name) is not None:
            raise KeyError('Algorithm already exists: ' + algo_name)

        self.__factory[algo_name] = {
            FUNC_CREATE_ALGORITHM: func_create_algo,
            FUNC_GET_ALGORITHM_PARAMS: func_get_algo_params
        }

    def create(self, algo_name, algo_params, **kwargs):
        """Creates the algorithm with the specified name and parameters.

        This function raises a KeyError when the name of the algorithm
        is not registered for this algorithm factory.

        Args:
            algo_name(str): name of the algorithm.
            algo_params(dict): parameters of the algorithm as name-value pairs.

        Keyword Args:
            num_threads(int): the max number of threads the algorithm can use.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.
        """
        if self.__factory.get(algo_name) is None:
            raise KeyError('Algorithm does not exist: ' + algo_name)

        func_create_algo = self.__factory[algo_name][FUNC_CREATE_ALGORITHM]
        algo = func_create_algo(dict(algo_params), **kwargs)
        algo.name = algo_name

        return algo

    def get_api_name(self):
        """Gets the name of the API associated with the factory.

        Returns:
            api_name(str) name of the API.
        """
        return self.__api_name

    def get_algorithm_params(self, algo_name):
        """Gets the parameters for the specified algorithm name.

        This function raises a KeyError when the name of the algorithm
        is not registered for this algorithm factory.

        Args:
            algo_name(str): name of the algorithm.

        Returns:
            algo_params(ConfigParameters): the parameters of the algorithm.
        """
        if algo_name not in self.__factory:
            raise KeyError('Algorithm does not exist: ' + algo_name)

        return self.__factory[algo_name][FUNC_GET_ALGORITHM_PARAMS]()

    def get_available(self):
        """Gets the available algorithms in the factory.

        Returns:
            algo_list(array like): dict entries with algorithm name and params.
        """
        algo_list = []

        for algo_name, entry in self.__factory.items():
            algo_params = entry[FUNC_GET_ALGORITHM_PARAMS]()
            algo_list.append({
                ALGORITHM_NAME: algo_name,
                ALGORITHM_PARAMS: algo_params.to_dict()
            })

        return algo_list

    def get_available_algorithm_names(self):
        """Gets the names of the available algorithms in the factory.

        Returns:
            algorithm_names(array like): list algorithm names.
        """
        algorithm_names = []

        for algo_name, _ in self.__factory.items():
            algorithm_names.append(algo_name)

        return algorithm_names

    def get_entries(self):
        """Gets the algorithm entries of the factory.

        Returns:
            factory entries(dict)
        """
        return dict(self.__factory)


def create_algorithm_factory_from_list(api_name, algo_list):
    """Creates an AlgorithmFactory from a list of tuples.

    Each tuple consists of:

    1) algorithm name
    2) algorithm creation function
    3) algorithm params function

    Args:
        api_name(str): name of the API associated with the factory.
        algo_list(array like): list of tuples.

    Returns:
        (AlgorithmFactory) filled with the specified tuples.
    """
    factory = AlgorithmFactory(api_name)

    for _, algo in enumerate(algo_list):
        (algo_name, func_create_algo, func_get_algo_params) = algo
        factory.add(
            algo_name,
            func_create_algo,
            func_get_algo_params
        )

    return factory
