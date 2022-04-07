"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from ..common import *
from .pipeline import ModelPipeline


class PredictorPipeline(ModelPipeline):

    def test_model_ratings(self, model, output_path, batch_size, **kwargs):
        start_index = 0
        while start_index < len(self.test_set):
            user_batch = self.test_set[start_index : start_index + batch_size]
            predictions = model.predict_batch(user_batch)
            predictions.to_csv(output_path, mode='a', sep='\t', header=False, index=False)
            start_index += batch_size


def create_predictor_pipeline_lenskit():
    api, factory = get_lenskit_predictor_factory()
    return PredictorPipeline(api, factory)


def create_predictor_pipeline_surprise():
    api, factory = get_surprise_predictor_factory()
    return PredictorPipeline(api, factory)
