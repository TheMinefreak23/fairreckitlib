"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time

from .pipeline import ModelPipeline


class RecommenderPipeline(ModelPipeline):

    def test_model(self, model, model_folder, callback, **kwargs):
        callback.on_begin_test_model(model, self.test_set)
        start = time.time()

        file_path = model_folder + self.stringify_model(model) + '.tsv'
        test_users = self.test_set.user.unique()

        batch_size = 1000
        start_index = 0
        total = len(test_users)
        print(total)
        while start_index < len(test_users):
            user_batch = test_users[start_index: start_index + batch_size]
            total -= len(user_batch)
            recs = model.recommend_batch(user_batch, num_items=kwargs['num_items'])
            recs.to_csv(file_path, mode='a', sep='\t', header=False, index=False)
            start_index += batch_size
            print(total)

        end = time.time()
        callback.on_end_test_model(model, self.test_set, end - start)