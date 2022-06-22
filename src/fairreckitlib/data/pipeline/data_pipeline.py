"""This module contains functionality of the complete data pipeline.

Classes:

    DataPipeline: class that performs dataset operations in preparation for the model pipeline.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time
from typing import Callable, Optional, Tuple

import pandas as pd

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.events.event_error import ON_FAILURE_ERROR, ErrorEventArgs
from ...core.io.io_create import create_dir
from ...core.pipeline.core_pipeline import CorePipeline
from ..data_transition import DataTransition
from ..filter.filter_config import DataSubsetConfig
from ..filter.filter_constants import KEY_DATA_SUBSET
from ..filter.filter_event import FilterDataframeEventArgs
from ..filter.filter_passes import filter_from_filter_passes
from ..ratings.convert_config import ConvertConfig
from ..ratings.convert_event import ConvertRatingsEventArgs
from ..ratings.rating_converter_factory import KEY_RATING_CONVERTER
from ..set.dataset import Dataset
# from ..filter.filter_constants import KEY_DATA_FILTERS, deduce_filter_type
from ..split.split_config import SplitConfig
from ..split.split_constants import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from ..split.split_event import SplitDataframeEventArgs
from .data_config import DataMatrixConfig
from .data_event import ON_BEGIN_DATA_PIPELINE, ON_END_DATA_PIPELINE, DatasetEventArgs
from .data_event import ON_BEGIN_LOAD_DATASET, ON_END_LOAD_DATASET, DatasetMatrixEventArgs
from .data_event import ON_BEGIN_FILTER_DATASET, ON_END_FILTER_DATASET
from .data_event import ON_BEGIN_CONVERT_RATINGS, ON_END_CONVERT_RATINGS
from .data_event import ON_BEGIN_SPLIT_DATASET, ON_END_SPLIT_DATASET
from .data_event import ON_BEGIN_SAVE_SETS, ON_END_SAVE_SETS, SaveSetsEventArgs


class DataPipeline(CorePipeline):
    """Data Pipeline to prepare a dataset for a transition to the ModelPipeline(s).

    The pipeline is intended to be reused multiple times depending on the specified
    datasets. This is not limited to using a dataset only once as they are numbered
    internally to distinguish them later.
    For each dataset the following steps are performed in order:

    1) create output directory.
    2) load the dataset into a dataframe.
    3) filter rows based on 'user'/'item' columns. (optional)
    4) convert 'rating' column. (optional)
    5) split the dataframe into a train and test set.
    6) save the train and test set in the output directory.

    Public methods:

    run
    """

    def __init__(self, data_factory: GroupFactory, event_dispatcher: EventDispatcher):
        """Construct the DataPipeline.

        Args:
            data_factory: the factory with available data modifier factories.
            event_dispatcher: used to dispatch data/IO events when running the pipeline.
        """
        CorePipeline.__init__(self, event_dispatcher)
        self.split_datasets = {}
        self.data_factory = data_factory

    def run(self,
            output_dir: str,
            dataset: Dataset,
            data_config: DataMatrixConfig,
            is_running: Callable[[], bool]) -> Optional[DataTransition]:
        """Run the entire data pipeline from beginning to end.

        Args:
            output_dir: the path of the directory to store the output.
            dataset: the dataset to run the pipeline on.
            data_config: the dataset matrix configurations.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Raises:
            FileNotFoundError: when the dataset matrix file does not exist.
            RuntimeError: when any data modifiers are not found in their respective factories.

        Returns:
            the data transition output of the pipeline.
        """
        self.event_dispatcher.dispatch(DatasetEventArgs(
            ON_BEGIN_DATA_PIPELINE,
            dataset.get_name()
        ))

        start = time.time()

        # step 1
        data_dir = self.create_data_output_dir(output_dir, data_config)

        # step 2
        dataframe = self.load_from_dataset(dataset, data_config.matrix)
        if not is_running():
            return None

        # step 3
        # if data_config:

        dataframe = self.filter_rows(output_dir, dataframe, data_config)
        if not is_running():
            return None

        # step 4
        dataframe = self.convert_ratings(dataset,
                                         data_config.matrix,
                                         dataframe,
                                         data_config.converter)
        if not is_running():
            return None

        # step 5
        train_set, test_set = self.split(dataframe, data_config.splitting)
        if not is_running():
            return None

        # step 6
        train_set_path, test_set_path = self.save_sets(data_dir, train_set, test_set)

        end = time.time()

        self.event_dispatcher.dispatch(DatasetEventArgs(
            ON_END_DATA_PIPELINE,
            dataset.get_name()
        ), elapsed_time=end - start)

        data_output = DataTransition(
            dataset,
            data_config.matrix,
            data_dir,
            train_set_path,
            test_set_path,
            (dataframe['rating'].min(), dataframe['rating'].max())
        )

        return data_output

    def create_data_output_dir(self, output_dir: str, data_config: DataMatrixConfig) -> str:
        """Create the data output directory for a dataset.

        Args:
            output_dir: the path of the directory to store the output.
            data_config: the dataset matrix configuration to create a directory for.

        Returns:
            the path of the directory where the output data can be stored.
        """
        dataset_matrix_name = data_config.dataset + '_' + data_config.matrix
        if not self.split_datasets.get(dataset_matrix_name):
            self.split_datasets[dataset_matrix_name] = 0

        index = self.split_datasets[dataset_matrix_name]
        self.split_datasets[dataset_matrix_name] += 1

        data_dir = os.path.join(output_dir, dataset_matrix_name + '_' + str(index))
        return create_dir(data_dir, self.event_dispatcher)

    def load_from_dataset(self, dataset: Dataset, matrix_name: str) -> pd.DataFrame:
        """Load in the desired dataset matrix into a dataframe.

        The loaded dataframe contains at least three columns 'user', 'item', 'rating'.
        In addition, the 'timestamp' column can be present when available in the specified dataset.

        Args:
            dataset: the dataset to load a matrix dataframe from.
            matrix_name: the name of the matrix to load from the dataset.

        Raises:
            FileNotFoundError: when the dataset matrix file does not exist.

        Returns:
            the dataframe belonging to the specified dataset.
        """
        self.event_dispatcher.dispatch(DatasetMatrixEventArgs(
            ON_BEGIN_LOAD_DATASET,
            dataset.get_name(),
            matrix_name,
            dataset.get_matrix_file_path(matrix_name)
        ))

        start = time.time()

        try:
            dataframe = dataset.load_matrix(matrix_name)
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_FAILURE_ERROR,
                'Failure: to load dataset matrix ' + dataset.get_name() + '_' + matrix_name
            ))
            # raise again so the data run aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(DatasetMatrixEventArgs(
            ON_END_LOAD_DATASET,
            dataset.get_name(),
            matrix_name,
            dataset.get_matrix_file_path(matrix_name)
        ), elapsed_time=end - start)

        return dataframe

    def filter_rows(self,
                    output_dir: str,
                    dataframe: pd.DataFrame,
                    subset: DataSubsetConfig) -> pd.DataFrame:
        """Apply the specified subset filters to the dataframe.

        The subset is created by applying multiple filter passes to the dataframe individually.
        These filter passes are then combined to form the resulting dataframe.

        Args:
            dataframe: the dataframe to filter with at least two columns: 'user', 'item'.
            subset: the subset to create of to the dataframe.

        Returns:
            the dataframe with the specified subgroup filters applied to it.
        """
        # early exit, because no filtering is needed
        if len(subset.filter_passes) == 0:
            return dataframe

        self.event_dispatcher.dispatch(FilterDataframeEventArgs(
            ON_BEGIN_FILTER_DATASET,
            subset
        ))

        start = time.time()
        filter_factory = self.data_factory.get_factory(KEY_DATA_SUBSET)
        # TODO aggregated the set using the given filters
        dataframe = filter_from_filter_passes(self, output_dir, dataframe, subset, filter_factory)
        end = time.time()

        self.event_dispatcher.dispatch(FilterDataframeEventArgs(
            ON_END_FILTER_DATASET,
            subset
        ), elapsed_time=end - start)

        return dataframe

    def convert_ratings(self,
                        dataset: Dataset,
                        matrix_name: str,
                        dataframe: pd.DataFrame,
                        convert_config: ConvertConfig) -> pd.DataFrame:
        """Convert the ratings in the dataframe with the specified rating modifier.

        Args:
            dataset: the dataset to load the matrix and rating_type from.
            matrix_name: the name of the dataset matrix.
            dataframe: the dataframe to convert the ratings of.
                At the least a 'rating' column is expected to be present.
            convert_config: the configuration of the converter to apply to the 'rating' column.

        Raises:
            RuntimeError: when the converter specified by the configuration is not available.

        Returns:
            the converted dataframe or the input dataframe when no converter is specified.
        """
        if convert_config is None:
            return dataframe

        self.event_dispatcher.dispatch(ConvertRatingsEventArgs(
            ON_BEGIN_CONVERT_RATINGS,
            convert_config
        ))

        start = time.time()

        converter_factory = self.data_factory.get_factory(KEY_RATING_CONVERTER)
        dataset_converter_factory = converter_factory.get_factory(dataset.get_name())
        matrix_converter_factory = dataset_converter_factory.get_factory(matrix_name)

        converter = matrix_converter_factory.create(convert_config.name, convert_config.params)
        if converter is None:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_FAILURE_ERROR,
                'Failure: to get converter from factory: ' + convert_config.name
            ))
            # raise error so the data run aborts
            raise RuntimeError()

        dataframe = converter.run(dataframe)

        end = time.time()

        self.event_dispatcher.dispatch(ConvertRatingsEventArgs(
            ON_END_CONVERT_RATINGS,
            convert_config
        ), elapsed_time=end - start)

        return dataframe

    def split(self,
              dataframe: pd.DataFrame,
              split_config: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into a train and test set.

        This will be split 80/20 (or a similar ratio), and be done either random, or timestamp-wise.
        The dataframe is expected to have at least three columns: 'user', 'item', 'rating'.
        In addition, the 'timestamp' column is required for temporal splits.

        Args:
            dataframe: the dataframe to split into a train and test set.
            split_config: the dataset splitting configuration.

        Raises:
            RuntimeError: when the splitter specified by the configuration is not available.

        Returns:
            the train and test set split of the specified dataframe.
        """
        self.event_dispatcher.dispatch(SplitDataframeEventArgs(
            ON_BEGIN_SPLIT_DATASET,
            split_config
        ))

        start = time.time()
        split_kwargs = {KEY_SPLIT_TEST_RATIO: split_config.test_ratio}
        split_factory = self.data_factory.get_factory(KEY_SPLITTING)
        splitter = split_factory.create(split_config.name, split_config.params, **split_kwargs)
        if splitter is None:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_FAILURE_ERROR,
                'Failure: to get splitter from factory: ' + split_config.name
            ))
            # raise error so the data run aborts
            raise RuntimeError()

        train_set, test_set = splitter.run(dataframe)
        end = time.time()

        self.event_dispatcher.dispatch(SplitDataframeEventArgs(
            ON_END_SPLIT_DATASET,
            split_config
        ), elapsed_time=end - start)

        return train_set, test_set

    def save_sets(self,
                  output_dir: str,
                  train_set: pd.DataFrame,
                  test_set: pd.DataFrame) -> Tuple[str, str]:
        """Save the train and test sets to the desired output directory.

        Args:
            output_dir: the path of the directory to store both sets.
            train_set: the train set to save with at least three columns: 'user', 'item', 'rating'.
            test_set: the test set to save with at least three columns: 'user', 'item', 'rating'.

        Returns:
            the paths where the train and test set are stored.
        """
        headers_to_save = ['user', 'item', 'rating']

        train_set = train_set[headers_to_save]
        test_set = test_set[headers_to_save]

        train_set_path = os.path.join(output_dir, 'train_set.tsv')
        test_set_path = os.path.join(output_dir, 'test_set.tsv')

        self.event_dispatcher.dispatch(SaveSetsEventArgs(
            ON_BEGIN_SAVE_SETS,
            train_set_path,
            test_set_path
        ))

        start = time.time()
        train_set.to_csv(train_set_path, sep='\t', header=False, index=False)
        test_set.to_csv(test_set_path, sep='\t', header=False, index=False)
        end = time.time()

        self.event_dispatcher.dispatch(SaveSetsEventArgs(
            ON_END_SAVE_SETS,
            train_set_path,
            test_set_path
        ), elapsed_time=end - start)

        return train_set_path, test_set_path
