"""This module contains all event ids, event args and a print switch for the data pipeline.

Constants:

    ON_BEGIN_DATA_PIPELINE: id of the event that is used when the data pipeline starts.
    ON_BEGIN_FILTER_DATASET: id of the event that is used when dataset filtering starts.
    ON_BEGIN_LOAD_DATASET: id of the event that is used when a dataset is being loaded.
    ON_BEGIN_MODIFY_DATASET: id of the event that is used when dataset ratings are being modified.
    ON_BEGIN_SAVE_SETS: id of the event that is used when the train and test sets are being saved.
    ON_BEGIN_SPLIT_DATASET: id of the event that is used when a dataset is being split.
    ON_END_DATA_PIPELINE: id of the event that is used when the data pipeline ends.
    ON_END_FILTER_DATASET: id of the event that is used when dataset filtering finishes.
    ON_END_LOAD_DATASET: id of the event that is used when a dataset has been loaded.
    ON_END_MODIFY_DATASET: id of the event that is used when dataset ratings have been modified.
    ON_END_SAVE_SETS: id of the event that is used when the train and test sets have been saved.
    ON_END_SPLIT_DATASET: id of the event that is used when a dataset has been split.

Classes:

    DatasetEventArgs: event args related to a dataset.
    DatasetMatrixEventArgs: event args related to a dataset matrix.
    SaveSetsEventArgs: event args related to saving a train and test set.

Functions:

    get_data_events: list of data pipeline event IDs.
    get_data_event_print_switch: switch to print data pipeline event arguments by ID.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from ...core.events.event_dispatcher import EventArgs
from ...core.io.event_io import DataframeEventArgs, print_load_df_event_args
from ..filter.filter_event import print_filter_event_args
from ..ratings.convert_event import print_convert_event_args
from ..split.split_event import print_split_event_args

ON_BEGIN_DATA_PIPELINE = 'DataPipeline.on_begin'
ON_BEGIN_FILTER_DATASET = 'DataPipeline.on_begin_filter_dataset'
ON_BEGIN_LOAD_DATASET = 'DataPipeline.on_begin_load_dataset'
ON_BEGIN_CONVERT_RATINGS = 'DataPipeline.on_begin_convert_ratings'
ON_BEGIN_SAVE_SETS = 'DataPipeline.on_begin_save_sets'
ON_BEGIN_SPLIT_DATASET = 'DataPipeline.on_begin_split_dataset'
ON_END_DATA_PIPELINE = 'DataPipeline.on_end'
ON_END_FILTER_DATASET = 'DataPipeline.on_end_filter_dataset'
ON_END_LOAD_DATASET = 'DataPipeline.on_end_load_dataset'
ON_END_CONVERT_RATINGS = 'DataPipeline.on_end_convert_ratings'
ON_END_SAVE_SETS = 'DataPipeline.on_end_save_sets'
ON_END_SPLIT_DATASET = 'DataPipeline.on_end_split_dataset'


@dataclass
class DatasetEventArgs(EventArgs):
    """Dataset Event Arguments.

    event_id: the unique ID that classifies the dataset event.
    dataset_name: the name of the dataset.
    """

    dataset_name: str


@dataclass
class DatasetMatrixEventArgs(DatasetEventArgs):
    """Dataset Matrix Event Arguments.

    event_id: the unique ID that classifies the dataset matrix event.
    dataset_name: the name of the dataset.
    matrix_name: the name of the dataset matrix.
    matrix_file_path: the path to the file of the dataset matrix.
    """

    matrix_name: str
    matrix_file_path: str


@dataclass
class SaveSetsEventArgs(EventArgs):
    """Save Sets Event Arguments.

    event_id: the unique ID that classifies the save sets event.
    train_set_path: the path to the file of the train set.
    test_set_path: the path to the file of the test set.
    """

    train_set_path: str
    test_set_path: str


def get_data_events() -> List[str]:
    """Get a list of data pipeline event IDs.

    Returns:
        a list of unique data pipeline event IDs.
    """
    return [
        # DatasetEventArgs
        ON_BEGIN_DATA_PIPELINE,
        ON_END_DATA_PIPELINE,
        # FilterDatasetEventArgs
        ON_END_FILTER_DATASET,
        ON_BEGIN_FILTER_DATASET,
        # DatasetMatrixEventArgs
        ON_BEGIN_LOAD_DATASET,
        ON_END_LOAD_DATASET,
        # ConvertRatingsEventArgs
        ON_BEGIN_CONVERT_RATINGS,
        ON_END_CONVERT_RATINGS,
        # SplitDataframeEventArgs
        ON_BEGIN_SPLIT_DATASET,
        ON_END_SPLIT_DATASET,
        # SaveSetsEventArgs
        ON_BEGIN_SAVE_SETS,
        ON_END_SAVE_SETS,
    ]


def get_data_event_print_switch(elapsed_time: float=None) -> Dict[str,Callable[[EventArgs], None]]:
    """Get a switch that prints data pipeline event IDs.

    Returns:
        the print data pipeline event switch.
    """
    return  {
        ON_BEGIN_DATA_PIPELINE:
            lambda args: print('\nStarting Data Pipeline:', args.dataset_name),
        ON_BEGIN_CONVERT_RATINGS: print_convert_event_args,
        ON_BEGIN_FILTER_DATASET: print_filter_event_args,
        ON_BEGIN_LOAD_DATASET:
            lambda args: print_load_df_event_args(DataframeEventArgs(
                args.event_id,
                args.matrix_file_path,
                'dataset matrix'
            )),
        ON_BEGIN_SAVE_SETS:
            lambda args: print('Saving train set to', args.train_set_path,
                               '\nSaving test set to', args.test_set_path),
        ON_BEGIN_SPLIT_DATASET: print_split_event_args,
        ON_END_DATA_PIPELINE:
            lambda args: print('Finished Data Pipeline:', args.dataset_name,
                               f'in {elapsed_time:1.4f}s'),
        ON_END_CONVERT_RATINGS:
            lambda args: print_convert_event_args(args, elapsed_time),
        ON_END_FILTER_DATASET:
            lambda args: print_filter_event_args(args, elapsed_time),
        ON_END_LOAD_DATASET:
            lambda args: print_load_df_event_args(DataframeEventArgs(
                args.event_id,
                args.matrix_file_path,
                'dataset matrix'
            ), elapsed_time=elapsed_time),
        ON_END_SAVE_SETS:
            lambda args: print(f'Saved train and test sets in {elapsed_time:1.4f}s'),
        ON_END_SPLIT_DATASET:
            lambda args: print_split_event_args(args, elapsed_time)
    }
