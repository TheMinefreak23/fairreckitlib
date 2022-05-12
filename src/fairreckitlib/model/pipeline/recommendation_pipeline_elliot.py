"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import numpy as np
import pandas as pd

from ...core.event_io import ON_MAKE_DIR, ON_REMOVE_DIR, ON_RENAME_FILE, ON_REMOVE_FILE
from ...data.utility import save_yml
from .model_pipeline import RATING_OUTPUT_FILE
from .recommendation_pipeline import RecommendationPipeline


class RecommendationPipelineElliot(RecommendationPipeline):
    """Recommendation Pipeline implementation for the Elliot framework."""

    def __init__(self, factory, event_dispatcher):
        RecommendationPipeline.__init__(self, factory, event_dispatcher)
        self.train_set_path = None
        self.test_set_path = None

    def load_train_and_test_set(self, train_set_path, test_set_path):
        # the loading is done by the Elliot framework, delay until after it is done.
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path

    def train_and_test_model(self, model, model_dir, is_running, **kwargs):
        params = model.get_params()
        params['meta'] = {'verbose': False, 'save_recs': True, 'save_weights': False}

        top_k = kwargs['num_items']

        temp_dir = self.__create_temp_dir(model_dir)
        yml_path = os.path.join(temp_dir, 'config.yml')

        data = {
            'experiment': {
                'dataset': 'df',
                'data_config': {
                    'strategy': 'fixed',
                    'train_path': os.path.join('..', '..', 'train_set.tsv'),
                    'test_path': os.path.join('..', '..', 'test_set.tsv'),
                },
                'top_k': top_k,
                'models': {
                    model.get_name(): params
                },
                'evaluation': {
                    'simple_metrics': ['Precision']
                },
                'path_output_rec_result': model_dir,
                'path_output_rec_weight': temp_dir,
                'path_output_rec_performance': temp_dir
            }
        }

        save_yml(yml_path, data)

        # stops the logo from being spammed to the console
        from elliot.run import run_experiment
        run_experiment(yml_path)

        self.__clear_temp_dir(temp_dir)
        if params.get('epochs'):
            # remove everything so that only the final epochs file remains
            self.__clear_unused_epochs(params['epochs'], model_dir)

        result_file_path = self.__reconstruct_rank_column(model_dir, top_k)

        # load the train and test set now the elliot framework is done
        RecommendationPipeline.load_train_and_test_set(
            self,
            self.train_set_path,
            self.test_set_path
        )

        return result_file_path


    def __create_temp_dir(self, model_dir):
        """Creates a temp directory to store unnecessary artifacts.

        Args:
            model_dir(str): the directory where the computed ratings can be stored.

        Returns:
            temp_dir(str): the path to the temp directory that was created.
        """
        temp_dir = os.path.join(model_dir, 'temp')
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
            self.event_dispatcher.dispatch(
                ON_MAKE_DIR,
                dir=temp_dir
            )

        return temp_dir

    def __clear_temp_dir(self, temp_dir):
        """Clears and removes the temp directory.

        Args:
            temp_dir(str): the path to the temp directory.
        """
        for file in os.listdir(temp_dir):
            file_name = os.fsdecode(file)
            file_path = os.path.join(temp_dir, file_name)
            if os.path.isdir(file_path):
                os.rmdir(file_path)
                self.event_dispatcher.dispatch(
                    ON_REMOVE_DIR,
                    dir=temp_dir
                )
            else:
                os.remove(file_path)
                self.event_dispatcher.dispatch(
                    ON_REMOVE_FILE,
                    file=file_path
                )

        os.rmdir(temp_dir)

    def __clear_unused_epochs(self, num_epochs, model_dir):
        """Clears unused epochs from the model output directory.

        Recommenders with an 'epochs' parameter will generate computed ratings
        for each epoch. Only the final epoch is needed.

        Args:
            num_epochs(int): the number of epochs that was run by the algorithm.
            model_dir(str): the directory where the computed ratings are stored.
        """
        used_epoch = 'it=' + str(num_epochs)
        for file in os.listdir(model_dir):
            file_name = os.fsdecode(file)
            # skip model settings json
            if 'settings.json' in file_name:
                continue

            file_path = os.path.join(model_dir, file_name)

            if used_epoch not in file_name:
                os.remove(file_path)
                self.event_dispatcher.dispatch(
                    ON_REMOVE_FILE,
                    file=file_path
                )

    def __reconstruct_rank_column(self, model_dir, top_k):
        """Reconstruct the rank column in the result file that the framework generated.

        Args:
            model_dir(str): the directory where the computed ratings are stored.
            top_k(int): the topK that was used to compute the ratings.

        Returns:
            result_file_path(str): the path to the computed ratings file.
        """

        result_file_path = self.__rename_result(model_dir)
        result = pd.read_csv(
            result_file_path,
            sep='\t',
            header=None,
            names=['user', 'item', 'score']
        )

        # create topK ranking array
        row_count = len(result)
        ranks = np.zeros(row_count)
        for i in range(row_count):
            ranks[i] = i % top_k + 1

        # add rank column
        result['rank'] = ranks
        result['rank'] = result['rank'].astype(int)

        # overwrite result
        result[['rank', 'user', 'item', 'score']].to_csv(
            result_file_path,
            sep='\t',
            header=True,
            index=False
        )

        return result_file_path

    def __rename_result(self, model_dir):
        """Renames the computed ratings file to be consistent.

        Args:
            model_dir(str): the directory where the computed ratings are stored.

        Returns:
            dst_path(str): the file path of the result after renaming.
        """
        for file in os.listdir(model_dir):
            file_name = os.fsdecode(file)
            # skip the model settings json
            if '.tsv' not in file_name:
                continue

            src_path = os.path.join(model_dir, file_name)
            dst_path = os.path.join(model_dir, RATING_OUTPUT_FILE)

            os.rename(src_path, dst_path)
            self.event_dispatcher.dispatch(
                ON_RENAME_FILE,
                src_file=src_path,
                dst_file=dst_path
            )

            return dst_path
