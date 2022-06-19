"""This module tests the model pipeline functionality.

Classes:

    DummyModelPipeline: dummy model pipeline to test not implemented errors.

Functions:

    test_model_pipeline_interface_errors: test interface errors for not implemented functions.
    test_model_pipeline_errors: test model pipeline for various errors that can be raised.
    test_model_pipeline_early_stop: test the early stopping of the model pipeline.
    test_run_model_pipelines: test the model pipeline (run) integration.
    assert_model_run_error: assert the model pipeline to run into an error.
    create_data_transition: create data transition for a dataset-matrix pair.
    create_model_type_config_coverage: create model configurations that cover each API.
    create_model_type_config_duplicates: create model configurations with duplicate models.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from src.fairreckitlib.core.config.config_factories import Factory, GroupFactory
from src.fairreckitlib.core.core_constants import \
    DEFAULT_TOP_K, KEY_RATED_ITEMS_FILTER, MODEL_RATINGS_FILE
from src.fairreckitlib.core.core_constants import VALID_TYPES, TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.core.core_constants import IMPLICIT_API, LENSKIT_API, SURPRISE_API
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_dir
from src.fairreckitlib.data.data_factory import create_data_factory
from src.fairreckitlib.data.data_transition import DataTransition
from src.fairreckitlib.data.ratings.convert_config import ConvertConfig
from src.fairreckitlib.data.ratings.convert_constants import CONVERTER_RANGE, RATING_TYPE_THRESHOLD
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.data.split.split_config import create_default_split_config
from src.fairreckitlib.data.pipeline.data_run import DataPipeline, DataMatrixConfig
from src.fairreckitlib.model.algorithms.implicit import implicit_algorithms
from src.fairreckitlib.model.algorithms.lenskit import lenskit_algorithms
from src.fairreckitlib.model.algorithms.surprise import surprise_algorithms
from src.fairreckitlib.model.algorithms.base_algorithm import BaseAlgorithm
from src.fairreckitlib.model.pipeline.model_config import ModelConfig
from src.fairreckitlib.model.pipeline.model_pipeline import ModelPipeline
from src.fairreckitlib.model.pipeline.model_run import ModelPipelineConfig, run_model_pipelines
from src.fairreckitlib.model.model_factory import create_model_factory
from .conftest import is_always_running
from .test_model_algorithm_matrices import MATRIX_DIR, MATRIX_FILE, MATRIX_RATING_SCALE
from .test_model_algorithms import DummyPredictor, DummyRecommender

model_kwargs = {'num_items': DEFAULT_TOP_K, 'num_threads': 1, KEY_RATED_ITEMS_FILTER: True}
unknown_data_transition = DataTransition(None, 'unknown', 'unknown', 'unknown', 'unknown', (0, 0))


class DummyModelPipeline(ModelPipeline):
    """Dummy model pipeline to test not implemented errors."""

    def load_test_set_users(self) -> None:
        """Raise NotImplementedError."""
        ModelPipeline.load_test_set_users(self)

    def test_model_ratings(
            self,
            model: BaseAlgorithm,
            user_batch: List[int],
            **kwargs) -> pd.DataFrame:
        """Raise NotImplementedError."""
        ModelPipeline.test_model_ratings(self, model, user_batch, **kwargs)


def test_model_pipeline_interface_errors(
        model_event_dispatcher: EventDispatcher) -> None:
    """Test model pipeline interface errors for not implemented functions."""
    model_pipeline = DummyModelPipeline(
        Factory('dummy'),
        unknown_data_transition,
        model_event_dispatcher
    )

    pytest.raises(NotImplementedError, model_pipeline.load_test_set_users)
    pytest.raises(NotImplementedError, model_pipeline.test_model_ratings, None, [])


@pytest.mark.parametrize('model_type', VALID_TYPES)
def test_model_pipeline_errors(
        io_tmp_dir: str,
        model_type: str,
        model_event_dispatcher: EventDispatcher) -> None:
    """Test model pipeline for various errors that can be raised."""
    model_type_factory = create_model_factory().get_factory(model_type)

    for api_name in model_type_factory.get_available_names():
        api_algo_factory = model_type_factory.get_factory(api_name)

        # create model pipeline with unknown train and test set files
        model_pipeline = api_algo_factory.create_pipeline(
            api_algo_factory,
            unknown_data_transition,
            model_event_dispatcher
        )

        print('\nModelPipeline: file not found errors\n')
        # test failure for file errors
        pytest.raises(FileNotFoundError, model_pipeline.load_train_set_matrix)
        pytest.raises(FileNotFoundError, model_pipeline.load_test_set_users)
        pytest.raises(FileNotFoundError, model_pipeline.load_train_set_dataframe)
        pytest.raises(FileNotFoundError, model_pipeline.load_test_set_dataframe)

        # reload model pipeline that loads the train and test set successfully
        model_pipeline = api_algo_factory.create_pipeline(
            api_algo_factory,
            DataTransition(None, 'user-movie-rating', MATRIX_DIR,
                           MATRIX_FILE, MATRIX_FILE, MATRIX_RATING_SCALE),
            model_event_dispatcher
        )

        print('\nModelPipeline: construction warning')
        # test warning for unknown algorithm
        assert not model_pipeline.run('', [ModelConfig('unknown', {})], is_always_running), \
            'did not expect model directory to be returned for an unknown model'

        model_error_tuples = [
            (ArithmeticError, 'ArithmeticError'),
            (MemoryError, 'MemoryError'),
            (RuntimeError, 'RuntimeError'),
        ]
        # test errors for construction, training and testing algorithms
        for error, error_name in model_error_tuples:
            if model_type == TYPE_PREDICTION:
                api_algo_factory.add_obj(error_name, DummyPredictor, None)
            elif model_type == TYPE_RECOMMENDATION:
                api_algo_factory.add_obj(error_name, DummyRecommender, None)
            else:
                break

            model_config = ModelConfig(error_name, {})
            model_kwargs['const_error'] = error

            print('\nModelPipeline: construction error =>', error_name)
            assert_model_run_error(io_tmp_dir, model_config, model_pipeline, **model_kwargs)

            del model_kwargs['const_error']
            model_kwargs['train_error'] = error

            print('\nModelPipeline: training error =>', error_name)
            assert_model_run_error(io_tmp_dir, model_config, model_pipeline, **model_kwargs)

            del model_kwargs['train_error']
            model_kwargs['test_error'] = error

            print('\nModelPipeline: testing error =>', error_name)
            assert_model_run_error(io_tmp_dir, model_config, model_pipeline, **model_kwargs)


@pytest.mark.parametrize('_model_type', VALID_TYPES)
def test_model_pipeline_early_stop(_model_type: str) -> None:
    """Test the early stopping return statements of the model pipeline."""
    # TODO test model pipeline early stopping


@pytest.mark.parametrize('model_type', VALID_TYPES)
def test_run_model_pipelines(
        data_registry: DataRegistry,
        io_tmp_dir: str,
        model_type: str,
        model_event_dispatcher: EventDispatcher) -> None:
    """Test the model pipeline (run) integration."""
    model_type_factory = create_model_factory().get_factory(model_type)

    # test catching of the FileNotFoundError in the model pipeline
    assert not run_model_pipelines(
        ModelPipelineConfig(
            io_tmp_dir,
            unknown_data_transition,
            model_type_factory,
            {}
        ),
        model_event_dispatcher,
        is_always_running,
        **model_kwargs
    )

    # test failure for unknown algorithm API factory
    assert not run_model_pipelines(
        ModelPipelineConfig(
            io_tmp_dir,
            unknown_data_transition,
            model_type_factory,
            {'unknown': []}
        ),
        model_event_dispatcher,
        is_always_running,
        **model_kwargs
    )

    # test model pipeline for several dataset-matrix pairs and API models
    for dataset_name in data_registry.get_available_sets():
        if not 'Sample' in dataset_name:
            continue

        for matrix_name in data_registry.get_set(dataset_name).get_available_matrices():
            output_dir = os.path.join(io_tmp_dir, dataset_name + '_' + matrix_name)
            create_dir(output_dir, model_event_dispatcher)

            data_transition = create_data_transition(
                output_dir,
                data_registry,
                (dataset_name, matrix_name)
            )

            models_config, num_models = create_model_type_config_coverage(model_type_factory)
            model_dirs = run_model_pipelines(
                ModelPipelineConfig(
                    output_dir,
                    data_transition,
                    model_type_factory,
                    models_config
                ),
                model_event_dispatcher,
                is_always_running,
                **model_kwargs
            )

            assert len(model_dirs) == num_models, \
                'expected all models to be processed by the model pipeline'

            test_set = pd.read_table(
                data_transition.test_set_path, header=None, names=['user', 'item', 'rating']
            )
            for model_dir in model_dirs:
                rating_set = os.path.join(model_dir, MODEL_RATINGS_FILE)
                assert os.path.isfile(rating_set), 'expected rating file for model computation'

                rating_set = pd.read_table(rating_set)
                if 'score' in rating_set:
                    assert len(test_set['user'].unique()) * DEFAULT_TOP_K == len(rating_set), \
                        'expected item recommendations for every user in the test set'
                elif 'prediction' in rating_set:
                    assert len(test_set) == len(rating_set), \
                        'expected user-item prediction for every pair in the test set'


def assert_model_run_error(
        model_dir: str,
        model_config: ModelConfig,
        model_pipeline: ModelPipeline,
        **kwargs) -> None:
    """Assert the model pipeline to run into an error."""
    assert not model_pipeline.run(model_dir, [model_config], is_always_running, **kwargs), \
        'did not expect model directory to be returned for a model that raised an error'
    assert model_pipeline.tested_models[model_config.name] == 0, \
        'expected model counter to be initialized before the error was raised'
    assert not os.path.isdir(model_pipeline.get_model_output_dir(model_dir, model_config.name)), \
        'expected model directory to be removed after an error was raised'


def create_data_transition(
        data_dir: str,
        datasets_registry: DataRegistry,
        dataset_pair: Tuple[str, str]) -> DataTransition:
    """Create data transition for a dataset-matrix pair using the data pipeline."""
    data_pipeline = DataPipeline(create_data_factory(datasets_registry), EventDispatcher())

    return data_pipeline.run(
        data_dir,
        datasets_registry.get_set(dataset_pair[0]),
        DataMatrixConfig(
            dataset_pair[0],
            dataset_pair[1],
            [],
            ConvertConfig(CONVERTER_RANGE, {'upper_bound': RATING_TYPE_THRESHOLD}),
            create_default_split_config()
        ),
        is_always_running
    )


def create_model_type_config_coverage(
        model_type_factory: GroupFactory) -> Tuple[Dict[str, List[ModelConfig]], int]:
    """Create several model configuration that covers each API from the model type factory."""
    models_config = {}
    num_models = 0

    for api_name in model_type_factory.get_available_names():
        api_factory = model_type_factory.get_factory(api_name)
        api_algo_names = {
            IMPLICIT_API: [
                implicit_algorithms.ALTERNATING_LEAST_SQUARES,
                implicit_algorithms.LOGISTIC_MATRIX_FACTORIZATION,
            ],
            LENSKIT_API: [
                lenskit_algorithms.POP_SCORE,
                lenskit_algorithms.RANDOM,
            ],
            SURPRISE_API: [
                surprise_algorithms.NORMAL_PREDICTOR,
                surprise_algorithms.SVD,
            ]
        }.get(api_name, [])

        models_config[api_name] = []
        for algo_name in api_algo_names:
            num_models += 1
            models_config[api_name].append(
                ModelConfig(
                    algo_name,
                    api_factory.create_params(algo_name).get_defaults()
                ))

    # lenskit_algorithm.RANDOM is not a predictor
    # and therefore also tests model discarding for prediction
    if model_type_factory.get_name() == TYPE_PREDICTION:
        num_models -= 1

    return models_config, num_models


def create_model_type_config_duplicates(
        model_type_factory: GroupFactory,
        num_duplicates: int=2) -> Dict[str, List[ModelConfig]]:
    """Create model configuration with duplicate models of one algorithm."""
    lenskit_model = lenskit_algorithms.POP_SCORE # both predictor and recommender
    lenskit_factory = model_type_factory.get_factory(LENSKIT_API)

    models_config = {LENSKIT_API: []}
    for _ in range(num_duplicates):
        models_config[LENSKIT_API].append(ModelConfig(
            lenskit_model,
            lenskit_factory.create_params(lenskit_model).get_defaults()
        ))

    return models_config
