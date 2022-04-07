"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import yaml

from ..common import *
from .recommender import RecommenderPipeline


class RecommenderPipelineElliot(RecommenderPipeline):

    def __init__(self, api_name, factory):
        RecommenderPipeline.__init__(self, api_name, factory)
        self.train_path = None
        self.test_path = None

    def __create_temp_folder(self, model_folder, callback):
        temp_folder = os.path.join(model_folder, 'temp')
        if not os.path.isdir(temp_folder):
            callback.on_create_folder(temp_folder)
            os.mkdir(temp_folder)

        return temp_folder

    def __clear_temp_folder(self, temp_folder):
        for file in os.listdir(temp_folder):
            file_name = os.fsdecode(file)
            file_path = os.path.join(temp_folder, file_name)
            if os.path.isdir(file_path):
                os.rmdir(file_path)
            else:
                os.remove(file_path)

        os.rmdir(temp_folder)

    def __clear_unused_epochs(self, num_epochs, model_folder):
        used_epoch = 'it=' + str(num_epochs)
        for file in os.listdir(model_folder):
            file_name = os.fsdecode(file)
            file_path = os.path.join(model_folder, file_name)
            if used_epoch not in file_name:
                os.remove(file_path)

    def __rename_result(self, model_folder):
        for file in os.listdir(model_folder):
            file_name = os.fsdecode(file)
            src_path = os.path.join(model_folder, file_name)
            dst_path = os.path.join(model_folder, 'ratings.tsv')
            os.rename(src_path, dst_path)


    def load_train_test_set(self, train_set_path, test_set_path, callback):
        self.train_path = os.path.join('..', '..', 'train_set.tsv')
        self.test_path = os.path.join('..', '..', 'test_set.tsv')

    def train_test_model(self, model, model_folder, callback, **kwargs):
        params = dict(model.params)
        params['meta'] = {'verbose': True, 'save_recs': True, 'save_weights': False}

        temp_folder = self.__create_temp_folder(model_folder, callback)
        yml_path = os.path.join(temp_folder, 'config.yml')

        data = {
            'experiment': {
                'dataset': self.dataset_name,
                'data_config': {
                    'strategy': 'fixed',
                    'train_path': self.train_path,
                    'test_path': self.test_path,
                },
                'top_k': kwargs['num_items'],
                'models': {
                    model.name: params
                },
                'evaluation': {
                    'simple_metrics': ['Precision']
                },
                'path_output_rec_result': model_folder,
                'path_output_rec_weight': temp_folder,
                'path_output_rec_performance': temp_folder
            }
        }

        with open(yml_path, 'w') as file:
            yaml.dump(data, file)

        # stops the elliot logo from being spammed to the console
        from elliot.run import run_experiment
        run_experiment(yml_path)

        self.__clear_temp_folder(temp_folder)
        if params.get('epochs'):
            self.__clear_unused_epochs(params['epochs'], model_folder)

        self.__rename_result(model_folder)


def create_recommender_pipeline_elliot():
    api, factory = get_elliot_recommender_factory()
    return RecommenderPipelineElliot(api, factory)
