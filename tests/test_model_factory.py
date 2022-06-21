"""This module tests the model (type/API) factories.

Functions:

    test_model_factory: test algorithms and factories to be derived from the correct base class.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.config.config_factories import Factory, GroupFactory
from src.fairreckitlib.core.core_constants import \
    KEY_RATED_ITEMS_FILTER, TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.data.set.dataset_config import DATASET_RATINGS_EXPLICIT
from src.fairreckitlib.model.algorithms.base_algorithm import BaseAlgorithm
from src.fairreckitlib.model.algorithms.base_predictor import BasePredictor
from src.fairreckitlib.model.algorithms.base_recommender import BaseRecommender
from src.fairreckitlib.model.model_factory import create_model_factory


def test_model_factory() -> None:
    """Test algorithms and factories in the factory to be derived from the correct base class."""
    model_factory = create_model_factory()
    assert isinstance(model_factory, GroupFactory), 'expected model group factory.'

    assert bool(model_factory.get_factory(TYPE_PREDICTION)), 'missing prediction models.'
    assert bool(model_factory.get_factory(TYPE_RECOMMENDATION)), 'missing recommender models.'

    for model_type in model_factory.get_available_names():
        model_type_factory = model_factory.get_factory(model_type)
        assert isinstance(model_type_factory, GroupFactory), 'expected API group factory.'

        # lenskit KNN algorithms need a rating_type
        algo_kwargs = {'num_threads': 1, 'rating_type': DATASET_RATINGS_EXPLICIT}
        if model_type ==TYPE_RECOMMENDATION:
            algo_kwargs[KEY_RATED_ITEMS_FILTER] = True

        for algo_api_name in model_type_factory.get_available_names():
            algo_api_factory = model_type_factory.get_factory(algo_api_name)
            assert isinstance(algo_api_factory, Factory), 'expected algorithm factory.'

            for algo_name in algo_api_factory.get_available_names():
                algo = algo_api_factory.create(algo_name, None, **algo_kwargs)

                assert isinstance(algo, BaseAlgorithm), 'expected base algorithm.'

                if model_type == TYPE_PREDICTION:
                    assert isinstance(algo, BasePredictor), 'expected base predictor.'
                elif model_type == TYPE_RECOMMENDATION:
                    assert isinstance(algo, BaseRecommender), 'expected base recommender.'
                else:
                    raise NotImplementedError('Unknown model type: ' + model_type)
