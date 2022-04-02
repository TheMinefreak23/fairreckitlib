""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class DataPipelineCallback(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def on_begin_pipeline(self):
        raise NotImplementedError()

    @abstractmethod
    def on_begin_load_df(self, file_name):
        raise NotImplementedError()

    @abstractmethod
    def on_end_load_df(self, elapsed_time):
        raise NotImplementedError

    @abstractmethod
    def on_begin_aggregate(self, **filters):
        raise NotImplementedError

    @abstractmethod
    def on_end_aggregate(self, elapsed_time):
        raise NotImplementedError

    @abstractmethod
    def on_begin_convert(self):
        raise NotImplementedError

    @abstractmethod
    def on_end_convert(self, elapsed_time):
        raise NotImplementedError

    @abstractmethod
    def on_begin_split(self, ratio):
        raise NotImplementedError

    @abstractmethod
    def on_end_split(self, elapsed_time):
        raise NotImplementedError

    @abstractmethod
    def on_saving_sets(self, file_path):
      raise NotImplementedError

    @abstractmethod
    def on_saved_sets(self, file_path):
        raise NotImplementedError

    @abstractmethod
    def on_end_pipeline(self):
        raise NotImplementedError


class DataPipelineConsole(DataPipelineCallback):

    def on_begin_pipeline(self):
        print('Starting Data Pipeline')

    def on_begin_load_df(self, file_name):
        print('Loading in the dataframe: ', file_name)

    def on_end_load_df(self, elapsed_time):
        print('Loaded in the dataframe in {0:1.4f}s' .format(elapsed_time))

    def on_begin_aggregate(self, filters):
        print('Making an aggregation of the dataframe using: ' + str(filters))

    def on_end_aggregate(self, elapsed_time):
        print('Aggregated the dataframe in {0:1.4f}s' .format(elapsed_time))

    def on_begin_convert(self):
        print('Converting ratings')

    def on_end_convert(self, elapsed_time):
        print('Converted ratings in {0:1.4f}s' .format(elapsed_time))

    def on_begin_split(self, ratio):
        print('Splitting the dataset: ' + str(ratio))

    def on_end_split(self, elapsed_time):
        print('Splitted the set in {0:1.4f}' .format(elapsed_time))

    def on_saving_sets(self, file_path):
        print('Saving train and test sets to: ' + file_path)

    def on_saved_sets(self, file_path, elapsed_time):
        print('Saved train and test sets to: ' + file_path + ' in {0:1.4f}' .format(elapsed_time))

    def on_end_pipeline(self, elapsed_time):
        print('Finished Data Pipeline in {0:1.4f} \n' .format(elapsed_time))
