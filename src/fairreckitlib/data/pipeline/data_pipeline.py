"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta
from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List
import pandas as pd

from ...core.event_dispatcher import EventDispatcher
from ...core.event_io import ON_MAKE_DIR
from ...core.factories import Factory
from ..set.dataset import Dataset
from ..split.split_factory import KEY_SPLITTING
from ..ratings.range_converter import RatingConverter
from .data_event import ON_BEGIN_DATA_PIPELINE, ON_END_DATA_PIPELINE
from .data_event import ON_BEGIN_LOAD_DATASET, ON_END_LOAD_DATASET
from .data_event import ON_BEGIN_FILTER_DATASET, ON_END_FILTER_DATASET
from .data_event import ON_BEGIN_MODIFY_DATASET, ON_END_MODIFY_DATASET
from .data_event import ON_BEGIN_SPLIT_DATASET, ON_END_SPLIT_DATASET
from .data_event import ON_BEGIN_SAVE_SETS, ON_END_SAVE_SETS

@dataclass
class DataTransition:
    """Data Transition struct to transfer pipeline data."""

    dataset : Dataset
    output_dir: str
    train_set_path: str
    test_set_path: str
    rating_scale: (float, float)
    rating_type: str


class DataPipeline(metaclass=ABCMeta):
    """Data Pipeline to prepare datasets to be used in the ModelPipeline(s).

    From loading in the required dataset(s) to aggregating them, converting the ratings,
    splitting it into train/test set, and saving these in the designated directory.

    Args:
        data_factory(GroupFactory):
        event_dispatcher(EventDispatcher): used to dispatch data/IO events
            when running the pipeline.
    """

    def __init__(self, data_factory: Factory, event_dispatcher: EventDispatcher):
        """Construct the DataPipeline.

        Args:
            data_factory: #TODO explain this
            event_dispatcher: #TODO explain this
        """
        self.split_datasets = {}
        self.data_factory = data_factory
        self.event_dispatcher = event_dispatcher

    def run(self, output_dir: str, dataset: Dataset, data_config: Dict[str, Any],
            is_running: function) -> DataTransition:
        """Run the entire data pipeline from beginning to end.

        1) load the dataset into a dataframe.
        2) filter rows based on 'user'/'item' columns. (optional)
        3) convert 'rating' column. (optional)
        4) split the dataframe into a train and test set.
        5) save the train and test set in the output directory.

        Args:
            output_dir: the path of the directory to store the output.
            dataset: the dataset to run the pipeline on.
            data_config: the dataset configuration.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Returns:
            data_output: the output of the pipeline.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_DATA_PIPELINE,
            dataset=dataset
        )

        start = time.time()

        # step 1
        data_dir = self.create_data_output_dir(output_dir, dataset)
        dataframe = self.load_from_dataset(dataset)
        if not is_running():
            return None

        # step 2
        dataframe = self.filter_rows(dataframe, data_config.prefilters)
        if not is_running():
            return None

        # step 3
        dataframe, rating_type = self.convert_ratings(dataset, dataframe,
                                                      data_config.rating_converter)
        if not is_running():
            return None

        # step 4
        train_set, test_set = self.split(dataframe, data_config.splitting)
        if not is_running():
            return None

        # step 5
        train_set_path, test_set_path = self.save_sets(data_dir, train_set, test_set)

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_DATA_PIPELINE,
            dataset=dataset,
            elapsed_time=end - start
        )

        data_output = DataTransition(
            dataset,
            data_dir,
            train_set_path,
            test_set_path,
            (dataframe['rating'].min(), dataframe['rating'].max()),
            rating_type
        )

        return data_output

    def create_data_output_dir(self, output_dir: str, dataset: Dataset) -> str:
        """Create the data output directory for a dataset.

        Args:
            output_dir: the path of the directory to store the output.
            dataset: the dataset to create a directory for.

        Returns:
            data_dir: the path of the directory where the output data can be stored.
        """
        if not self.split_datasets.get(dataset.name):
            self.split_datasets[dataset.name] = 0

        index = self.split_datasets[dataset.name]
        self.split_datasets[dataset.name] += 1

        data_dir = os.path.join(output_dir, dataset.name + '_' + str(index))
        os.mkdir(data_dir)
        self.event_dispatcher.dispatch(
            ON_MAKE_DIR,
            dir=data_dir
        )

        return data_dir

    def load_from_dataset(self, dataset: Dataset) -> pd.DataFrame:
        """Load in the desired dataset into a dataframe.

        Args:
            dataset: the dataset to load the matrix dataframe from.

        Returns:
            dataframe: belonging to the specified dataset. The
                dataframe contains at least three columns 'user', 'item', 'rating'.
                In addition, the 'timestamp' column can be present when
                available in the specified dataset.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_LOAD_DATASET,
            dataset=dataset
        )

        start = time.time()
        dataframe = dataset.load_matrix_df()
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_DATASET,
            dataset=dataset,
            elapsed_time=end - start
        )

        return dataframe

    def filter_rows(self, dataframe: pd.DataFrame, prefilters: List) -> pd.DataFrame:
        """Apply the specified filters to the dataframe.

        Args:
            dataframe: the dataset to filter with at least
                two columns: 'user', 'item'.
            prefilters: list of user/item filters to apply #TODO specify list content type
                to the dataframe.

        Returns:
            dataframe: with the specified filters applied to it.
        """
        # early exit, because no filtering is needed
        if len(prefilters) == 0:
            return dataframe

        self.event_dispatcher.dispatch(
            ON_BEGIN_FILTER_DATASET,
            prefilters=prefilters
        )

        start = time.time()
        # TODO aggregated the set using the given filters
        for data_filter in prefilters:
            continue
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_FILTER_DATASET,
            prefilters=prefilters,
            elapsed_time=end - start
        )

        return dataframe

    def convert_ratings(self, dataset: Dataset, dataframe: pd.DataFrame, 
                        rating_converter: RatingConverter) -> tuple[pd.DataFrame, str]:
        """Convert the ratings in the dataframe with the specified rating modifier.

        Args:
            dataset: the dataset to load the matrix and rating_type from.
            dataframe: the dataframe to convert the ratings of.
                At the least a 'rating' column is expected to be present.
            rating_converter: the converter to apply to the 'rating' column.

        Returns:
            dataframe: with the modified 'rating' column.
            rating_type: the new rating type, either implicit or explicit.
        """
        if rating_converter is None:
            return dataframe, dataset.get_matrix_info('rating_type')

        self.event_dispatcher.dispatch(
            ON_BEGIN_MODIFY_DATASET,
            rating_converter=rating_converter
        )

        start = time.time()
        dataframe, rating_type = rating_converter.run(dataframe)
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_MODIFY_DATASET,
            rating_converter=rating_converter,
            elapsed_time=end - start
        )

        return dataframe, rating_type

    def split(self, dataframe: pd.DataFrame, split_config: Dict[str, Any])\
              -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into a train and test set.

        This will be split 80/20 (or a similar ratio), and be done either random, or timestamp-wise.

        Args:
            dataframe: the dataset to split with at least
                three columns: 'user', 'item', 'rating'. In addition, the 'timestamp' column
                is required for temporal splits.
            split_config: the dataset splitting configuration.

        Returns:
            train_set: the train set split of the specified dataframe.
            test_set: the test set split of the specified dataframe.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_SPLIT_DATASET,
            split_type=split_config.type,
            test_ratio=split_config.test_ratio
        )

        start = time.time()
        split_factory = self.data_factory.get_factory(KEY_SPLITTING)
        splitter = split_factory.create(split_config.type, split_config.params)
        train_set, test_set = splitter.run(dataframe, split_config.test_ratio)
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_SPLIT_DATASET,
            split_type=split_config.type,
            test_ratio=split_config.test_ratio,
            elapsed_time=end - start
        )

        return train_set, test_set

    def save_sets(self, output_dir: str, train_set: pd.DataFrame, test_set: pd.DataFrame)\
                  -> tuple[str, str]:
        """Save the train and test sets to the desired output directory.

        Args:
            output_dir: the path of the directory to store both sets.
            train_set: the train set to save with at least
                three columns: 'user', 'item', 'rating'.
            test_set: the test set to save with at least
                three columns: 'user', 'item', 'rating'.

        Returns:
            train_set_path: the path where the train set was stored.
            test_set_path: the path where the test set was stored.
        """
        headers_to_save = ['user', 'item', 'rating']

        train_set = train_set[headers_to_save]
        test_set = test_set[headers_to_save]

        train_set_path = os.path.join(output_dir, 'train_set.tsv')
        test_set_path = os.path.join(output_dir, 'test_set.tsv')

        self.event_dispatcher.dispatch(
            ON_BEGIN_SAVE_SETS,
            train_set_path=train_set_path,
            test_set_path=test_set_path
        )

        start = time.time()
        train_set.to_csv(train_set_path, sep='\t', header=False, index=False)
        test_set.to_csv(test_set_path, sep='\t', header=False, index=False)
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_SAVE_SETS,
            train_set_path=train_set_path,
            test_set_path=test_set_path,
            elapsed_time=end - start
        )

        return train_set_path, test_set_path
