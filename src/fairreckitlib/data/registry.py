"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .config import DATASET_CONFIG_FILE
from .config import DATASET_PREFIX
from .processor.processor_lfm_1b import DataProcessorLFM1B
from .processor.processor_lfm_2b import DataProcessorLFM2B
from .processor.processor_lfm_360k import DataProcessorLFM360K
from .processor.processor_ml_100k import DataProcessorML100K
from .processor.processor_ml_25m import DataProcessorML25M
from .set import Dataset
from .utility import load_yml

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

    Args:
        data_dir(str): path to the directory that contains the datasets.
    """
    def __init__(self, data_dir):
        if not os.path.isdir(data_dir):
            raise IOError('Failed to initialize DataRegistry: '
                          'unknown data directory => ' + data_dir)

        self.__registry = {}
        self.processors = {
            DATASET_LFM_1B: DataProcessorLFM1B,
            DATASET_LFM_2B: DataProcessorLFM2B,
            DATASET_LFM_360K: DataProcessorLFM360K,
            DATASET_ML_100K: DataProcessorML100K,
            DATASET_ML_25M: DataProcessorML25M
        }

        for file in os.listdir(data_dir):
            file_name = os.fsdecode(file)
            dataset_dir = os.path.join(data_dir, file_name)
            # skip all entries that are not a directory
            if not os.path.isdir(dataset_dir):
                continue

            config_file_name = DATASET_PREFIX + file_name + DATASET_CONFIG_FILE
            config_file_path = os.path.join(dataset_dir, config_file_name)
            if not os.path.isfile(config_file_path):
                if self.processors.get(file_name) is None:
                    print('Unknown data processor:', file_name)
                    continue

                config = self.processors[file_name](file_name).run(dataset_dir, config_file_name)
            else:
                config = load_yml(config_file_path)

            self.__registry[file_name] = Dataset(file_name, dataset_dir, config)

    def get_available_processors(self):
        """Gets the names of the available processors in the registry.

        Returns:
            processor_names(array like): list data processor names.
        """
        processor_names = []

        for processor_name in self.processors:
            processor_names.append(processor_name)

        return processor_names

    def get_available_sets(self):
        """Gets the names of the available datasets in the registry.

        Returns:
            dataset_names(array like): list dataset names.
        """
        dataset_names = []

        for dataset_name in self.__registry:
            dataset_names.append(dataset_name)

        return dataset_names

    def get_info(self):
        """Gets the matrix information for each available dataset.

        Returns:
            (dict): where the key corresponds to the dataset name and
                the value corresponds to the matrix information dictionary.
        """
        info = {}

        for dataset_name, dataset in self.__registry.items():
            info[dataset_name] = dataset.get_matrix_info()

        return info

    def get_set(self, dataset_name):
        """Gets the dataset with the specified name.

        Args:
            dataset_name(str): name of the dataset to retrieve.

        Returns:
            (Dataset): the retrieved set or None when not present.
        """
        return self.__registry.get(dataset_name)
