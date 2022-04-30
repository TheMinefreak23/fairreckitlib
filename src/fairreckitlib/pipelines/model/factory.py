""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.algorithms.constants import ALGORITHM_FACTORY
from fairreckitlib.algorithms.elliot_alg.factory import get_elliot_recommender_factory
from fairreckitlib.algorithms.implicit_alg.factory import get_implicit_recommender_factory
from fairreckitlib.algorithms.lenskit_alg.factory import get_lenskit_predictor_factory
from fairreckitlib.algorithms.lenskit_alg.factory import get_lenskit_recommender_factory
from fairreckitlib.algorithms.surprise_alg.factory import get_surprise_predictor_factory
from fairreckitlib.algorithms.surprise_alg.factory import get_surprise_recommender_factory
from .predictor import PredictorPipeline
from .recommender import RecommenderPipeline
from .recommender_elliot import RecommenderPipelineElliot

FUNC_CREATE_MODEL_PIPELINE = 'f_create_model_pipeline'


class ModelFactory:
    """Model Factory with available algorithms and pipeline creation."""
    def __init__(self):
        self.__factory = {}

    def add(self, algo_factory, func_create_pipeline):
        """Adds an algorithm factory with a specific API.

        This function raises a KeyError when the API of the new
        algorithm factory is already registered for this model factory.

        Args:
            algo_factory(AlgorithmFactory): factory of available algorithms.
            func_create_pipeline(function): ModelPipeline creation function that is
                utilizes the algo_factory to run computations.

        """
        api_name = algo_factory.get_api_name()
        if self.__factory.get(api_name) is not None:
            raise KeyError('API already exists: ' + api_name)

        self.__factory[api_name] = {
            ALGORITHM_FACTORY: algo_factory,
            FUNC_CREATE_MODEL_PIPELINE: func_create_pipeline
        }

    def create_pipeline(self, api_name, event_dispatcher):
        """Create a ModelPipeline for the specified API.

        This function raises a KeyError when the requested API
        is not registered for this factory.

        Args:
            api_name(str): name of the API to create a pipeline for.
            event_dispatcher(EventDispatcher): used to dispatch model/IO events
                when running the pipeline.

        Returns:
            (ModelPipeline) the requested pipeline associated with the API.
        """
        if self.__factory.get(api_name) is None:
            raise KeyError('API does not exist: ' + api_name)

        func_create = self.__factory[api_name][FUNC_CREATE_MODEL_PIPELINE]
        return func_create(
            api_name,
            self.__factory[api_name][ALGORITHM_FACTORY],
            event_dispatcher
        )

    def get_algorithm_factory(self, api_name):
        """Gets the algorithm factory for the specified API name.

        Args:
            api_name(str): name of the API to get the factory from.

        Returns:
            algo_factory(AlgorithmFactory): the requested API factory.
        """
        if api_name not in self.__factory:
            return None

        return self.__factory[api_name][ALGORITHM_FACTORY]

    def get_available_algorithms(self):
        """Gets the available algorithms in the factory for each API.

        Returns:
            (dict) with all algorithms in the factory.
                Each key-value pair describes an API:
                    key(str): name of the API.
                    value(array like): dict entries with algorithm name and params.
        """
        result = {}

        for api_name, api_dict in self.__factory.items():
            result[api_name] = api_dict[ALGORITHM_FACTORY].get_available()

        return result

    def get_available_api_names(self):
        """Gets the names of the available APIs in the factory.

        Returns:
            api_names(array like): list API names.
        """
        api_names = []

        for api_name, _ in self.__factory.items():
            api_names.append(api_name)

        return api_names


def create_model_factory_from_api_tuples(api_tuples):
    """Creates a ModelFactory from a list of API factory/pipeline tuples.

    Args:
        api_tuples(array like): list of tuples.
            Each tuple describes:
                func_get_api_factory: function that returns api_factory(AlgorithmFactory).
                func_create_pipeline: function to create a ModelPipeline for the API.

    Returns:
        (ModelFactory) filled with the specified API factory/pipeline entries.
    """
    factory = ModelFactory()

    for _, (func_get_api_factory, func_create_pipeline) in enumerate(api_tuples):
        factory.add(
            func_get_api_factory(),
            func_create_pipeline
        )

    return factory


def create_predictor_model_factory():
    """Creates a ModelFactory with all predictor algorithms.

    Consists of algorithms from two APIs:
        1) LensKit predictor algorithms.
        2) Surprise predictor algorithms.

    Returns:
        (ModelFactory) with all predictors.
    """
    return create_model_factory_from_api_tuples([
        (get_lenskit_predictor_factory, PredictorPipeline),
        (get_surprise_predictor_factory, PredictorPipeline)
    ])

def create_recommender_model_factory():
    """Creates a ModelFactory with all recommender algorithms.

    Consists of algorithms from three APIs:
        1) Elliot recommender algorithms.
        2) LensKit recommender algorithms.
        3) Implicit recommender algorithms.
        4) Surprise recommender algorithms.

    Returns:
        (ModelFactory) with all recommenders.
    """
    return create_model_factory_from_api_tuples([
        (get_elliot_recommender_factory, RecommenderPipelineElliot),
        (get_lenskit_recommender_factory, RecommenderPipeline),
        (get_implicit_recommender_factory, RecommenderPipeline),
        (get_surprise_recommender_factory, RecommenderPipeline)
    ])
