"""This module contains functionality of the complete data pipeline.

Classes:

    DataPipeline: class that performs dataset operations in preparation for the model pipeline.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta
import os
import time
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_FAILURE_ERROR
from ...core.event_io import ON_MAKE_DIR
from ...core.factories import GroupFactory
from ..data_transition import DataTransition
from ..set.dataset import Dataset
from ..filter.filter_constants import KEY_DATA_FILTERS
from ..split.split_config import SplitConfig
from ..split.split_constants import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from ..ratings.convert_config import ConvertConfig
from ..ratings.rating_converter_factory import KEY_RATING_CONVERTER
from .data_config import DatasetConfig
from .data_event import ON_BEGIN_DATA_PIPELINE, ON_END_DATA_PIPELINE
from .data_event import ON_BEGIN_LOAD_DATASET, ON_END_LOAD_DATASET
from .data_event import ON_BEGIN_FILTER_DATASET, ON_END_FILTER_DATASET
from .data_event import ON_BEGIN_MODIFY_DATASET, ON_END_MODIFY_DATASET
from .data_event import ON_BEGIN_SPLIT_DATASET, ON_END_SPLIT_DATASET
from .data_event import ON_BEGIN_SAVE_SETS, ON_END_SAVE_SETS


class DataPipeline(metaclass=ABCMeta):
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
        self.split_datasets = {}
        self.data_factory = data_factory
        self.event_dispatcher = event_dispatcher

    def run(self,
            output_dir: str,
            dataset: Dataset,
            data_config: DatasetConfig,
            is_running: Callable[[], bool]) -> Optional[DataTransition]:
        """Run the entire data pipeline from beginning to end.

        Two errors can be raised during execution of the pipeline:
        FileNotFoundError is raised if the dataset matrix file does not exist.
        RuntimeError is raised if any data modifiers are not found in their respective factories.

        Args:
            output_dir: the path of the directory to store the output.
            dataset: the dataset to run the pipeline on.
            data_config: the dataset configuration.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Returns:
            the data transition output of the pipeline.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_DATA_PIPELINE,
            dataset=dataset
        )

        start = time.time()

        # step 1
        data_dir = self.create_data_output_dir(output_dir, dataset)

        # step 2
        dataframe = self.load_from_dataset(dataset)
        if not is_running():
            return None

        # step 3
        dataframe = self.filter_rows(dataframe, data_config.prefilters)
        if not is_running():
            return None

        # step 4
        dataframe, rating_type = self.convert_ratings(dataset, dataframe, data_config.converter)
        if not is_running():
            return None

        # step 5
        train_set, test_set = self.split(dataframe, data_config.splitting)
        if not is_running():
            return None

        # step 6
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
            the path of the directory where the output data can be stored.
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
        """Load in the desired dataset matrix into a dataframe.

        It raises a FileNotFoundError when the dataset matrix file does not exist.

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

        try:
            dataframe = dataset.load_matrix_df()
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to load dataset matrix ' + dataset.name
            )
            # raise again so the data run aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_DATASET,
            dataset=dataset,
            elapsed_time=end - start
        )

        return dataframe

    def filter_rows(self, dataframe: pd.DataFrame, prefilters: List[Any]) -> pd.DataFrame:
        """Apply the specified filters to the dataframe.

        Args:
            dataframe: the dataset to filter with at least
                two columns: 'user', 'item'.
            prefilters: list of user/item filters to apply to the dataframe.

        Returns:
            the dataframe with the specified filters applied to it.
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

        filter_factory = self.data_factory.get_factory(KEY_DATA_FILTERS)

        for prefilter in prefilters:
            filterer = filter_factory.create(prefilter.name, prefilter.value)
            dataframe = filterer.run(dataframe)


        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_FILTER_DATASET,
            prefilters=prefilters,
            elapsed_time=end - start
        )

        return dataframe

    def convert_ratings(self,
                        dataset: Dataset,
                        dataframe: pd.DataFrame,
                        convert_config: ConvertConfig) -> Tuple[pd.DataFrame, str]:
        """Convert the ratings in the dataframe with the specified rating modifier.

        It raises a RuntimeError when the converter specified by the configuration is not available.

        Args:
            dataset: the dataset to load the matrix and rating_type from.
            dataframe: the dataframe to convert the ratings of.
                At the least a 'rating' column is expected to be present.
            convert_config: the configuration of the converter to apply to the 'rating' column.

        Returns:
            the converted dataframe and the type of rating, either 'explicit' or 'implicit'.
        """
        if convert_config is None:
            return dataframe, dataset.get_matrix_info('rating_type')

        self.event_dispatcher.dispatch(
            ON_BEGIN_MODIFY_DATASET,
            rating_converter=convert_config.name
        )

        start = time.time()
        convert_factory = self.data_factory.get_factory(KEY_RATING_CONVERTER)
        converter = convert_factory.create(convert_config.name, convert_config.params)
        if converter is None:
            self.event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to get converter from factory: ' + convert_config.name
            )
            # raise error so the data run aborts
            raise RuntimeError()

        dataframe, rating_type = converter.run(dataframe)
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_MODIFY_DATASET,
            rating_converter=convert_config.name,
            elapsed_time=end - start
        )

        return dataframe, rating_type

    def split(self,
              dataframe: pd.DataFrame,
              split_config: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into a train and test set.

        This will be split 80/20 (or a similar ratio), and be done either random, or timestamp-wise.
        It raises a RuntimeError when the splitter specified by the configuration is not available.

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
            split_name=split_config.name,
            test_ratio=split_config.test_ratio
        )

        start = time.time()
        split_kwargs = {KEY_SPLIT_TEST_RATIO: split_config.test_ratio}
        split_factory = self.data_factory.get_factory(KEY_SPLITTING)
        splitter = split_factory.create(split_config.name, split_config.params, **split_kwargs)
        if splitter is None:
            self.event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to get splitter from factory: ' + split_config.name
            )
            # raise error so the data run aborts
            raise RuntimeError()

        train_set, test_set = splitter.run(dataframe)
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_SPLIT_DATASET,
            split_name=split_config.name,
            test_ratio=split_config.test_ratio,
            elapsed_time=end - start
        )

        return train_set, test_set

    def save_sets(self,
                  output_dir: str,
                  train_set: pd.DataFrame,
                  test_set: pd.DataFrame) -> Tuple[str, str]:
        """Save the train and test sets to the desired output directory.

        Args:
            output_dir: the path of the directory to store both sets.
            train_set: the train set to save with at least
                three columns: 'user', 'item', 'rating'.
            test_set: the test set to save with at least
                three columns: 'user', 'item', 'rating'.

        Returns:
            the paths where the train and test set are stored.
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
