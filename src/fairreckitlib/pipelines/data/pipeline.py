"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta
from dataclasses import dataclass
import os
import time
from typing import Any

from fairreckitlib.data.set import Dataset
from fairreckitlib.events import data_event
from fairreckitlib.events import io_event


@dataclass
class SplitConfig:
    """Dataset Splitting Configuration."""

    test_ratio: float
    type: str
    params: {str: Any}


@dataclass
class DatasetConfig:
    """Dataset Configuration."""

    name: str
    prefilters: []
    rating_modifier: None
    splitting: SplitConfig


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
        split_factory(SplitFactory): factory of available splitters.
        event_dispatcher(EventDispatcher): used to dispatch data/IO events
            when running the pipeline.
    """
    def __init__(self, split_factory, event_dispatcher):
        self.split_datasets = {}
        self.split_factory = split_factory
        self.event_dispatcher = event_dispatcher

    def run(self, output_dir, dataset, data_config, is_running):
        """Runs the entire data pipeline from beginning to end.

        1) load the dataset into a dataframe.
        2) filter rows based on 'user'/'item' columns. (optional)
        3) convert 'rating' column. (optional)
        4) split the dataframe into a train and test set.
        5) save the train and test set in the output directory.

        Args:
            output_dir(str): the path of the directory to store the output.
            dataset(Dataset): the dataset to run the pipeline on.
            data_config(DatasetConfig): the dataset configuration.
            is_running(func -> bool): function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Returns:
            data_output(DataTransition): the output of the pipeline.
        """
        self.event_dispatcher.dispatch(
            data_event.ON_BEGIN_DATA_PIPELINE,
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
        dataframe = self.convert_ratings(dataframe, data_config.rating_modifier)
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
            data_event.ON_END_DATA_PIPELINE,
            dataset=dataset,
            elapsed_time=end - start
        )

        data_output = DataTransition(
            dataset,
            data_dir,
            train_set_path,
            test_set_path,
            (dataframe['rating'].min(), dataframe['rating'].max()),
            dataset.get_matrix_info('rating_type') # TODO this needs to be dynamically retrieved after modifying ratings
        )

        return data_output

    def create_data_output_dir(self, output_dir, dataset):
        """Creates the data output directory for a dataset.

        Args:
            output_dir(str): the path of the directory to store the output.
            dataset(Dataset): the dataset to create a directory for.

        Returns:
            data_dir(str): the path of the directory where the output data can be stored.
        """
        if not self.split_datasets.get(dataset.name):
            self.split_datasets[dataset.name] = 0

        index = self.split_datasets[dataset.name]
        self.split_datasets[dataset.name] += 1

        data_dir = os.path.join(output_dir, dataset.name + '_' + str(index))
        os.mkdir(data_dir)
        self.event_dispatcher.dispatch(
            io_event.ON_MAKE_DIR,
            dir=data_dir
        )

        return data_dir

    def load_from_dataset(self, dataset):
        """Loads in the desired dataset into a dataframe.

        Args:
            dataset(Dataset): the dataset to load the matrix dataframe from.

        Returns:
            dataframe(pandas.DataFrame): belonging to the specified dataset. The
                dataframe contains at least three columns 'user', 'item', 'rating'.
                In addition, the 'timestamp' column can be present when
                available in the specified dataset.
        """
        self.event_dispatcher.dispatch(
            data_event.ON_BEGIN_LOAD_DATASET,
            dataset=dataset
        )

        start = time.time()
        dataframe = dataset.load_matrix_df()
        end = time.time()

        self.event_dispatcher.dispatch(
            data_event.ON_END_LOAD_DATASET,
            dataset=dataset,
            elapsed_time=end - start
        )

        return dataframe

    def filter_rows(self, dataframe, prefilters):
        """Applies the specified filters to the dataframe.

        Args:
            dataframe(pandas.DataFrame): the dataset to filter with at least
                two columns: 'user', 'item'.
            prefilters(array like): list of user/item filters to apply
                to the dataframe.

        Returns:
            dataframe(pandas.DataFrame): with the specified filters applied to it.
        """
        # early exit, because no filtering is needed
        if len(prefilters) == 0:
            return dataframe

        self.event_dispatcher.dispatch(
            data_event.ON_BEGIN_FILTER_DATASET,
            prefilters=prefilters
        )

        start = time.time()
        # TODO aggregated the set using the given filters
        for data_filter in prefilters:
            continue
        end = time.time()

        self.event_dispatcher.dispatch(
            data_event.ON_END_FILTER_DATASET,
            prefilters=prefilters,
            elapsed_time=end - start
        )

        return dataframe

    def convert_ratings(self, dataframe, rating_modifier):
        """Converts the ratings in the dataframe with the specified rating modifier.

        Args:
            dataframe(pandas.DataFrame): the dataset to convert the ratings for.
                At the least a 'rating' column is expected to be present.
            rating_modifier(object): the modifier to apply to the 'rating' column.

        Returns:
            dataframe(pandas.DataFrame): with the modified 'rating' column.
        """
        if rating_modifier is None:
            return dataframe

        self.event_dispatcher.dispatch(
            data_event.ON_BEGIN_MODIFY_DATASET,
            rating_modifier=rating_modifier
        )

        start = time.time()
        # TODO convert the ratings of the dataset
        end = time.time()

        self.event_dispatcher.dispatch(
            data_event.ON_END_MODIFY_DATASET,
            rating_modifier=rating_modifier,
            elapsed_time=end - start
        )

        return dataframe

    def split(self, dataframe, split_config):
        """Splits the dataframe into a train and test set.

        This will be split 80/20 (or a similar ratio), and be done either random, or timestamp-wise.

        Args:
            dataframe(pandas.DataFrame): the dataset to split with at least
                three columns: 'user', 'item', 'rating'. In addition, the 'timestamp' column
                is required for temporal splits.
            split_config(SplitConfig): the dataset splitting configuration.

        Returns:
            train_set(pandas.DataFrame): the train set split of the specified dataframe.
            test_set(pandas.DataFrame): the test set split of the specified dataframe.
        """
        self.event_dispatcher.dispatch(
            data_event.ON_BEGIN_SPLIT_DATASET,
            split_type=split_config.type,
            test_ratio=split_config.test_ratio
        )

        start = time.time()
        splitter = self.split_factory.create(split_config.type, split_config.params)
        train_set, test_set = splitter.run(dataframe, split_config.test_ratio)
        end = time.time()

        self.event_dispatcher.dispatch(
            data_event.ON_END_SPLIT_DATASET,
            split_type=split_config.type,
            test_ratio=split_config.test_ratio,
            elapsed_time=end - start
        )

        return train_set, test_set

    def save_sets(self, output_dir, train_set, test_set):
        """Saves the train and test sets to the desired output directory.

        Args:
            output_dir(str): the path of the directory to store both sets.
            train_set(pandas.DataFrame): the train set to save with at least
                three columns: 'user', 'item', 'rating'.
            test_set(pandas.DataFrame): the test set to save with at least
                three columns: 'user', 'item', 'rating'.

        Returns:
            train_set_path(str): the path where the train set was stored.
            test_set_path(str): the path where the test set was stored.
        """

        headers_to_save = ['user', 'item', 'rating']

        train_set = train_set[headers_to_save]
        test_set = test_set[headers_to_save]

        train_set_path = os.path.join(output_dir, 'train_set.tsv')
        test_set_path = os.path.join(output_dir, 'test_set.tsv')

        self.event_dispatcher.dispatch(
            data_event.ON_BEGIN_SAVE_SETS,
            train_set_path=train_set_path,
            test_set_path=test_set_path
        )

        start = time.time()
        train_set.to_csv(train_set_path, sep='\t', header=False, index=False)
        test_set.to_csv(test_set_path, sep='\t', header=False, index=False)
        end = time.time()

        self.event_dispatcher.dispatch(
            data_event.ON_END_SAVE_SETS,
            train_set_path=train_set_path,
            test_set_path=test_set_path,
            elapsed_time=end - start
        )

        return train_set_path, test_set_path
