"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Callable

import numpy as np
import pandas as pd

from ...core.config_constants import MODEL_RATINGS_FILE
from ...core.event_dispatcher import EventDispatcher
from ...core.event_io import ON_MAKE_DIR, ON_REMOVE_DIR, ON_RENAME_FILE, ON_REMOVE_FILE
from ...core.factories import Factory
from ...data.utility import save_yml
from ..algorithms.elliot.elliot_recommender import ElliotRecommender
from .recommendation_pipeline import RecommendationPipeline


class RecommendationPipelineElliot(RecommendationPipeline):
    """Recommendation Pipeline implementation for the Elliot framework."""

    def __init__(self, factory: Factory, event_dispatcher: EventDispatcher):
        """Construct the elliot recommendation pipeline.

        Args:
            factory: factory of available elliot recommenders.
            event_dispatcher: used to dispatch model/IO events when running the pipeline.
        """
        RecommendationPipeline.__init__(self, factory, event_dispatcher)
        self.train_set_path = None
        self.test_set_path = None

    def load_train_and_test_set(self, train_set_path: str, test_set_path: str) -> None:
        """Load the train and test set that all models can use.

        The loading is done by the Elliot framework, delay until after it is done.

        Args:
            train_set_path: path to where the train set is stored.
            test_set_path: path to where the test set is stored.
        """
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path

    def train_and_test_model(self,
                             model: ElliotRecommender,
                             model_dir: str,
                             is_running: Callable[[], bool],
                             **kwargs) -> str:
        """Train and test the specified model.

        Convert the model configuration into a yml file that is accepted by the framework.
        Feed it to the framework to obtain results, clear unwanted artifacts and modify the
        ratings file so that it conforms to the standard convention.

        Args:
            model: the model that needs to be trained.
            model_dir: the path of the directory where the computed ratings can be stored.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce.

        Returns:
            the path to the file where the model's computed ratings are stored.
        """
        params = model.get_params()
        params['meta'] = {'verbose': False, 'save_recs': True, 'save_weights': False}

        top_k = kwargs['num_items']

        temp_dir = self.create_temp_dir(model_dir)
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

        self.clear_temp_dir(temp_dir)
        if params.get('epochs'):
            # remove everything so that only the final epochs file remains
            self.clear_unused_epochs(params['epochs'], model_dir)

        result_file_path = self.reconstruct_rank_column(model_dir, top_k)
        if not is_running():
            return result_file_path

        # load the train and test set now the elliot framework is done
        RecommendationPipeline.load_train_and_test_set(
            self,
            self.train_set_path,
            self.test_set_path
        )

        return result_file_path


    def create_temp_dir(self, model_dir: str) -> str:
        """Create a temp directory to store unnecessary artifacts.

        Args:
            model_dir: the directory where the computed ratings can be stored.

        Returns:
            the path to the temp directory that was created.
        """
        temp_dir = os.path.join(model_dir, 'temp')
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
            self.event_dispatcher.dispatch(
                ON_MAKE_DIR,
                dir=temp_dir
            )

        return temp_dir

    def clear_temp_dir(self, temp_dir: str) -> None:
        """Clear and remove the temp directory.

        Args:
            temp_dir: the path to the temp directory.
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

    def clear_unused_epochs(self, num_epochs: int, model_dir: str) -> None:
        """Clear unused epochs from the model output directory.

        Recommenders with an 'epochs' parameter will generate computed ratings
        for each epoch. Only the final epoch is needed.

        Args:
            num_epochs: the number of epochs that was run by the algorithm.
            model_dir: the directory where the computed ratings are stored.
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

    def reconstruct_rank_column(self, model_dir: str, top_k: int) -> str:
        """Reconstruct the rank column in the result file that the framework generated.

        Args:
            model_dir: the directory where the computed ratings are stored.
            top_k: the topK that was used to compute the ratings.

        Returns:
            the path to the computed ratings file.
        """
        result_file_path = self.rename_result(model_dir)
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

    def rename_result(self, model_dir: str) -> str:
        """Rename the computed ratings file to be consistent with other pipelines.

        Args:
            model_dir: the directory where the computed ratings are stored.

        Returns:
            the file path of the result after renaming.
        """
        for file in os.listdir(model_dir):
            file_name = os.fsdecode(file)
            # skip the model settings json
            if '.tsv' not in file_name:
                continue

            src_path = os.path.join(model_dir, file_name)
            dst_path = os.path.join(model_dir, MODEL_RATINGS_FILE)

            os.rename(src_path, dst_path)
            self.event_dispatcher.dispatch(
                ON_RENAME_FILE,
                src_file=src_path,
                dst_file=dst_path
            )

            return dst_path
