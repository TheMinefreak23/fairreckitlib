"""This module contains the base functionality shared by all dataset processors.

Classes:

    DatasetProcessorBase: the base class for dataset processors.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from ..dataset_config import DatasetConfig, DatasetMatrixConfig, DatasetTableConfig


class DatasetProcessorBase(metaclass=ABCMeta):
    """DataProcessor base class for all FairRecKit datasets.

    Datasets are preprocessed so that they will be of a recognized standard format
    on the other side. A configuration file is produced in the resulting dataset
    directory that stores the metadata for achieving this. For further information
    it is advised to take a look at the Dataset(Config) class.

    The dataset configuration mainly consists of:

    1) event tables: contain user event tables that can be used to construct a matrix of.
    2) matrix tables: contain available matrices associated with the dataset.
    3) (other) tables: contain the shared tables associated with the dataset.

    For each of these three categories an abstract function is exposed in order to retrieve
    (table name, table configuration processor) tuples. The tables names are expected to be
    unique across all categories. The table configuration processors are allowed to return
    None on failure and will be excluded from the final configuration. Moreover, tables that
    do not contain any records are excluded as well.
    The base dataset processor handles the processing logic. It needs to produce at least
    one valid event table or one valid matrix configuration to be successful, concluding
    that remaining tables are optional.

    Abstract methods:

    get_event_configs
    get_matrix_configs
    get_table_configs

    Public methods:

    run
    """

    def __init__(self, dataset_dir: str, dataset_name: str):
        """Construct the base DatasetProcessor.

        Args:
            dataset_name: path of the dataset directory.
            dataset_name: name of the dataset (processor).
        """
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

    @abstractmethod
    def get_event_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get event table configuration processors.

        Returns:
            a list of tuples consisting of the event table name and the event table processor.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_matrix_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetMatrixConfig]]]]:
        """Get matrix configuration processors.

        Returns:
            a list of tuples consisting of the matrix name and the matrix processor.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Returns:
            a list of tuples consisting of the table name and the table processor.
        """
        raise NotImplementedError()

    def run_event_table_processors(self) -> Dict[str, DatasetTableConfig]:
        """Run the dataset's event table processors.

        Returns:
            a dictionary with valid event table name-configuration pairs.
        """
        dataset_events = {}
        for table_name, process_config in self.get_event_configs():
            config = process_config()
            if config is not None and config.num_records > 0:
                dataset_events[table_name] = config

        return dataset_events

    def run_matrix_table_processors(self) -> Dict[str, DatasetMatrixConfig]:
        """Run the dataset's matrix processors.

        Returns:
            a dictionary with valid matrix name-configuration pairs.
        """
        dataset_matrices = {}
        for matrix_name, process_config in self.get_matrix_configs():
            config = process_config()
            if config is not None and config.table.num_records > 0:
                dataset_matrices[matrix_name] = config

        return dataset_matrices

    def run_table_processors(self) -> Dict[str, DatasetTableConfig]:
        """Run the dataset's additional table processors.

        Returns:
            a dictionary with valid table name-configuration pairs.
        """
        dataset_tables = {}
        for table_name, process_config in self.get_table_configs():
            config = process_config()
            if config is not None and config.num_records > 0:
                dataset_tables[table_name] = config

        return dataset_tables

    def run(self) -> Optional[DatasetConfig]:
        """Run the dataset configuration processor.

        Returns:
            the dataset configuration or None on failure.
        """
        dataset_events = self.run_event_table_processors()
        dataset_matrices = self.run_matrix_table_processors()
        if len(dataset_events) == 0 and len(dataset_matrices) == 0:
            return None

        return DatasetConfig(
            self.dataset_name,
            dataset_events,
            dataset_matrices,
            self.run_table_processors()
        )
