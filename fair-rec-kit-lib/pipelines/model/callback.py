"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class ModelPipelineCallback(metaclass=ABCMeta):

    @abstractmethod
    def on_begin_pipeline(self, api_name):
        raise NotImplementedError()

    @abstractmethod
    def on_begin_load_train_set(self, file_path):
        raise NotImplementedError()

    @abstractmethod
    def on_end_load_train_set(self, file_path, train_set, elapsed_time):
        raise NotImplementedError()

    @abstractmethod
    def on_begin_load_test_set(self, file_path):
        raise NotImplementedError()

    @abstractmethod
    def on_end_load_test_set(self, file_path, test_set, elapsed_time):
        raise NotImplementedError()

    @abstractmethod
    def on_begin_model(self, model_name):
        raise NotImplementedError()

    @abstractmethod
    def on_create_folder(self, folder_path):
        raise NotImplementedError()

    @abstractmethod
    def on_create_model(self, model_name, params):
        raise NotImplementedError()

    @abstractmethod
    def on_begin_train_model(self, model, train_set):
        raise NotImplementedError()

    @abstractmethod
    def on_end_train_model(self, model, train_set, elapsed_time):
        raise NotImplementedError()

    @abstractmethod
    def on_begin_test_model(self, model, test_set):
        raise NotImplementedError()

    @abstractmethod
    def on_end_test_model(self, model, test_set, elapsed_time):
        raise NotImplementedError()

    @abstractmethod
    def on_end_model(self, model_name):
        raise NotImplementedError()

    @abstractmethod
    def on_end_pipeline(self, api_name):
        raise NotImplementedError()


class ModelPipelineConsole(ModelPipelineCallback):

    def on_begin_pipeline(self, api_name):
        print('Starting Model Pipeline:', api_name)
        print('')

    def on_begin_load_train_set(self, file_path):
        print('Loading train set from', file_path)

    def on_end_load_train_set(self, file_path, train_set, elapsed_time):
        print('Loaded train set in {0:1.4f}s'.format(elapsed_time))
        print('')

    def on_begin_load_test_set(self, file_path):
        print('Loading test set from', file_path)

    def on_end_load_test_set(self, file_path, train_set, elapsed_time):
        print('Loaded test set in {0:1.4f}s'.format(elapsed_time))
        print('')

    def on_begin_model(self, model_name):
        print('Starting model:', model_name)

    def on_create_folder(self, folder_path):
        print('Creating folder:', folder_path)

    def on_create_model(self, model_name, params):
        print('Creating model', model_name)
        print('')

        for key in params:
            print(key, '=', params[key])

        print('')

    def on_begin_train_model(self, model, train_set):
        print('Training model...')

    def on_end_train_model(self, model, train_set, elapsed_time):
        print('Trained model in {0:1.4f}s'.format(elapsed_time))
        print('')

    def on_begin_test_model(self, model, test_set):
        print('Testing model...')

    def on_end_test_model(self, model, test_set, elapsed_time):
        print('Tested model in {0:1.4f}s'.format(elapsed_time))
        print('')

    def on_end_model(self, model_name):
        print('Finished model:', model_name)
        print('')

    def on_end_pipeline(self, api_name):
        print('Finished Model Pipeline:', api_name)
        print('')
