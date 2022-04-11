"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .set import Dataset


class DataRegistry:

    def __init__(self, data_dir):
        if not os.path.isdir(data_dir):
            raise IOError('Failed to initialize DataRegistry: '
                          'unknown data directory => ' + data_dir)

        self.__registry = dict()

        for file in os.listdir(data_dir):
            file_name = os.fsdecode(file)
            dataset_dir = os.path.join(data_dir, file_name)
            # skip all entries that are not a directory
            if not os.path.isdir(dataset_dir):
                continue  # break

            dataset_path = os.path.join(dataset_dir, file_name + '.tsv')
            if not os.path.isfile(dataset_path):
                # preprocess dataset to standardized format and generate metadata
                # the data.loader module should implement this functionality
                raise NotImplementedError()

            # TODO load in metadata and pass this to the Dataset constructor
            self.__registry[file_name] = Dataset(file_name, dataset_dir)

    def get_info(self):
        info = {}
        for dataset_name in self.__registry:
            info[dataset_name] = self.__registry[dataset_name].get_info()

        return info

    def get_set(self, dataset_name):
        return self.__registry[dataset_name]
