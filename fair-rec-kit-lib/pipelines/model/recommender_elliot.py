"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import yaml

from .recommender import RecommenderPipeline


class RecommenderPipelineElliot(RecommenderPipeline):

    def __init__(self, api_name, factory):
        RecommenderPipeline.__init__(self, api_name, factory)
        self.train_set_path = None
        self.test_set_path = None

    def __create_temp_folder(self, model_folder, callback):
        temp_folder = model_folder + 'temp/'
        if not os.path.isdir(temp_folder):
            callback.on_create_folder(temp_folder)
            os.mkdir(temp_folder)

        return temp_folder

    def load_train_test_set(self, train_set_path, test_set_path, callback):
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path

    def train_test_model(self, model, model_folder, callback, **kwargs):
        params = dict(model.get_params())
        params['meta'] = {'verbose': True, 'save_recs': True, 'save_weights': False}

        temp_folder = self.__create_temp_folder(model_folder, callback)
        yml_path = temp_folder + 'config.yml'

        data = {
            'experiment': {
                'dataset': self.dataset_name,
                'data_config': {
                    'strategy': 'fixed',
                    'train_path': '../../../' + self.train_set_path,
                    'test_path': '../../../' + self.test_set_path,
                },
                'top_k': kwargs['num_items'],
                'models': {
                    model.name: params
                },
                'gpu': 1,
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
