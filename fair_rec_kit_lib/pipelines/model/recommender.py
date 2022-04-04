"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time

from ..common import *
from .pipeline import ModelPipeline


class RecommenderPipeline(ModelPipeline):

    def test_model(self, model, model_folder, callback, **kwargs):
        callback.on_begin_test_model(model, self.test_set)
        start = time.time()

        file_path = os.path.join(model_folder, self.stringify_model(model) + '.tsv')
        test_users = self.test_set.user.unique()

        batch_size = 10000
        start_index = 0
        while start_index < len(test_users):
            user_batch = test_users[start_index: start_index + batch_size]
            recs = model.recommend_batch(user_batch, num_items=kwargs['num_items'])
            recs.to_csv(file_path, mode='a', sep='\t', header=False, index=False)
            start_index += batch_size

        end = time.time()
        callback.on_end_test_model(model, self.test_set, end - start)


def create_recommender_pipeline_implicit():
    api, factory = get_implicit_recommender_factory()
    return RecommenderPipeline(api, factory)


def create_recommender_pipeline_lenskit():
    api, factory = get_lenskit_recommender_factory()
    return RecommenderPipeline(api, factory)
