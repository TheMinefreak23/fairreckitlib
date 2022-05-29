"""This module contains all event ids, event args and a print switch for the model pipeline.

Constants:

    ON_BEGIN_LOAD_TEST_SET: id of the event that is used when a test set is being loaded.
    ON_BEGIN_LOAD_TRAIN_SET: id of the event that is used when a train set is being loaded.
    ON_BEGIN_MODEL_PIPELINE: id of the event that is used when the model pipeline starts.
    ON_BEGIN_TEST_MODEL: id of the event that is used when testing a model started.
    ON_BEGIN_TRAIN_MODEL: id of the event that is used when training a model started.
    ON_BEGIN_MODEL: id of the event that is used when a model computation started.
    ON_END_LOAD_TEST_SET: id of the event that is used when a test set has been loaded.
    ON_END_LOAD_TRAIN_SET: id of the event that is used when a train set has been loaded.
    ON_END_MODEL_PIPELINE: id of the event that is used when the model pipeline ends.
    ON_END_TEST_MODEL: id of the event that is used when testing a model finishes.
    ON_END_TRAIN_MODEL: id of the event that is used when training a model finishes.
    ON_END_MODEL: id of the event that is used when a model computation finishes.

Classes:

    ModelPipelineEventArgs: event args related to the model pipeline.
    ModelEventArgs: event args related to a model.

Functions:

    get_model_events: list of model pipeline event IDs.
    get_model_event_print_switch: switch to print model pipeline event arguments by ID.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from ...core.events.event_dispatcher import EventArgs
from ...core.events.event_io import print_load_df_event_args
from .model_config import ModelConfig

ON_BEGIN_LOAD_TEST_SET = 'ModelPipeline.on_begin_load_test_set'
ON_BEGIN_LOAD_TRAIN_SET = 'ModelPipeline.on_begin_load_train_set'
ON_BEGIN_MODEL_PIPELINE = 'ModelPipeline.on_begin'
ON_BEGIN_TEST_MODEL = 'ModelPipeline.on_begin_model_test'
ON_BEGIN_TRAIN_MODEL = 'ModelPipeline.on_begin_model_train'
ON_BEGIN_MODEL = 'ModelPipeline.on_begin_model'
ON_END_LOAD_TEST_SET = 'ModelPipeline.on_end_load_test_set'
ON_END_LOAD_TRAIN_SET = 'ModelPipeline.on_end_load_train_set'
ON_END_MODEL_PIPELINE = 'ModelPipeline.on_end'
ON_END_TEST_MODEL = 'ModelPipeline.on_end_test_model'
ON_END_TRAIN_MODEL = 'ModelPipeline.on_end_train_model'
ON_END_MODEL = 'ModelPipeline.on_end_model'


@dataclass
class ModelPipelineEventArgs(EventArgs):
    """Model Pipeline Event Arguments.

    event_id: the unique ID that classifies the model pipeline event.
    api_name: the name of the api that is used in the model pipeline.
    models_config: list of model configurations that is used in the model pipeline.
    """

    api_name: str
    models_config: List[ModelConfig]


@dataclass
class ModelEventArgs(EventArgs):
    """Model Event Arguments.

    event_id: the unique ID that classifies the model event.
    model_name: the name of the model.
    model_params: the parameters of the model.
    """

    model_name: str
    model_params: Dict[str, Any]


def get_model_events() -> List[str]:
    """Get a list of model pipeline event IDs.

    Returns:
        a list of unique model pipeline event IDs.
    """
    return [
        # DataframeEventArgs
        ON_BEGIN_LOAD_TEST_SET,
        ON_END_LOAD_TEST_SET,
        # DataframeEventArgs
        ON_BEGIN_LOAD_TRAIN_SET,
        ON_END_LOAD_TRAIN_SET,
        # ModelPipelineEventArgs
        ON_BEGIN_MODEL_PIPELINE,
        ON_END_MODEL_PIPELINE,
        # ModelEventArgs
        ON_BEGIN_TEST_MODEL,
        ON_END_TEST_MODEL,
        # ModelEventArgs
        ON_BEGIN_TRAIN_MODEL,
        ON_END_TRAIN_MODEL,
        # ModelEventArgs
        ON_BEGIN_MODEL,
        ON_END_MODEL
    ]


def get_model_event_print_switch(elapsed_time: float=None) -> Dict[str,Callable[[EventArgs],None]]:
    """Get a switch that prints model pipeline event IDs.

    Returns:
        the print model pipeline event switch.
    """
    return {
        ON_BEGIN_LOAD_TEST_SET: print_load_df_event_args,
        ON_BEGIN_LOAD_TRAIN_SET: print_load_df_event_args,
        ON_BEGIN_MODEL_PIPELINE:
            lambda args: print('\nStarting Model Pipeline:', args.api_name,
                                'to process', len(args.models_config), 'model(s)'),
        ON_BEGIN_MODEL:
            lambda args: print('Starting model:', args.model_name),
        ON_BEGIN_TEST_MODEL:
            lambda args: print('Testing model:', args.model_name),
        ON_BEGIN_TRAIN_MODEL:
            lambda args: print('Training model:', args.model_name),
        ON_END_LOAD_TEST_SET:
            lambda args: print_load_df_event_args(args, elapsed_time),
        ON_END_LOAD_TRAIN_SET:
            lambda args: print_load_df_event_args(args, elapsed_time),
        ON_END_MODEL_PIPELINE:
            lambda args: print('Finished Model Pipeline:', args.api_name,
                                f'in {elapsed_time:1.4f}s'),
        ON_END_MODEL:
            lambda args: print('Finished model:', args.model_name,
                                f'in {elapsed_time:1.4f}s'),
        ON_END_TEST_MODEL:
            lambda args: print(f'Tested model in {elapsed_time:1.4f}s'),
        ON_END_TRAIN_MODEL:
            lambda args: print(f'Trained model in {elapsed_time:1.4f}s'),
    }
