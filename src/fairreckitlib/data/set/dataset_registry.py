"""This module contains the data registry class.

Classes:

    DataRegistry: registry for available datasets after processing them into a standard format.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict, List, Optional

from ...core.io.io_utility import save_yml
from .dataset_config_parser import DatasetConfigParser
from .dataset_constants import DATASET_CONFIG_FILE
from .dataset import Dataset
from .processor.dataset_processor_lfm1b import DatasetProcessorLFM1B
from .processor.dataset_processor_lfm2b import DatasetProcessorLFM2B
from .processor.dataset_processor_lfm360k import DatasetProcessorLFM360K
from .processor.dataset_processor_ml100k import DatasetProcessorML100K
from .processor.dataset_processor_ml25m import DatasetProcessorML25M

DATASET_LFM_1B = 'LFM-1B'
DATASET_LFM_2B = 'LFM-2B'
DATASET_LFM_360K = 'LFM-360K'
DATASET_ML_100K = 'ML-100K'
DATASET_ML_25M = 'ML-25M'


class DataRegistry:
    """Data Registry with available datasets.

    The data directory is expected to exist or will raise an IOError.
    Each subdirectory is considered to store a single dataset. The name of
    the subdirectory needs to be exactly the same as one of the available
    processors to trigger automatic data processing.

    Public methods:

    get_available_processors
    get_available_sets
    get_info
    get_set
    """

    def __init__(self, data_dir: str, verbose: bool=True):
        """Construct the data registry and scan for available datasets.

        Args:
            data_dir: path to the directory that contains the datasets.
            verbose: whether the dataset parser should give verbose output.

        Raises:
            IOError: when the specified data directory does not exist.
        """
        if not os.path.isdir(data_dir):
            raise IOError('Unable to construct data registry from an unknown directory')

        self.registry = {}
        self.processors = {
            DATASET_LFM_1B: DatasetProcessorLFM1B,
            DATASET_LFM_2B: DatasetProcessorLFM2B,
            DATASET_LFM_360K: DatasetProcessorLFM360K,
            DATASET_ML_100K: DatasetProcessorML100K,
            DATASET_ML_25M: DatasetProcessorML25M
        }

        for file in os.listdir(data_dir):
            file_name = os.fsdecode(file)
            dataset_dir = os.path.join(data_dir, file_name)
            # skip all entries that are not a directory
            if not os.path.isdir(dataset_dir):
                continue

            config_file_path = os.path.join(dataset_dir, DATASET_CONFIG_FILE)
            if not os.path.isfile(config_file_path):
                if self.processors.get(file_name) is None:
                    print('Unknown dataset processor:', file_name)
                    continue

                config = self.processors[file_name](dataset_dir, file_name).run()
                if config is None:
                    print('Processing dataset failed:', file_name)
                    continue

                save_yml(config_file_path, config.to_yml_format())
            else:
                parser = DatasetConfigParser(verbose)
                config = parser.parse_dataset_config_from_yml(
                    dataset_dir,
                    DATASET_CONFIG_FILE,
                    self.get_available_sets()
                )
                if config is None:
                    print('Parsing dataset configuration failed:', file_name)
                    continue

            self.registry[config.dataset_name] = Dataset(dataset_dir, config)

    def get_available_processors(self) -> List[str]:
        """Get the names of the available processors in the registry.

        Returns:
            a list of data processor names.
        """
        processor_names = []

        for processor_name in self.processors:
            processor_names.append(processor_name)

        return processor_names

    def get_available_sets(self) -> List[str]:
        """Get the names of the available datasets in the registry.

        Returns:
            a list of dataset names.
        """
        dataset_names = []

        for dataset_name in self.registry:
            dataset_names.append(dataset_name)

        return dataset_names

    def get_info(self) -> Dict[str, Dict[str, Any]]:
        """Get the matrices' information for each available dataset.

        Returns:
            a dictionary where the key corresponds to the dataset name and
                the value corresponds to the matrices' information dictionary.
        """
        info = {}

        for dataset_name, dataset in self.registry.items():
            info[dataset_name] = dataset.get_matrices_info()

        return info

    def get_set(self, dataset_name: str) -> Optional[Dataset]:
        """Get the dataset with the specified name.

        Args:
            dataset_name: name of the dataset to retrieve.

        Returns:
            the retrieved set or None when not present.
        """
        return self.registry.get(dataset_name)
