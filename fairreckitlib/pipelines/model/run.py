"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .factory import get_predictor_pipeline_factory
from .factory import get_recommender_pipeline_factory


def run_predictor_model_pipelines(dataset, output_dir, train_path, test_path, rating_scale, rating_type, models_config, num_threads, callback):
    model_dirs = []

    for api_name in models_config:
        mp = get_predictor_pipeline_factory()[api_name]()
        model_dirs += mp.run(
            dataset.name,
            train_path,
            test_path,
            rating_scale,
            rating_type,
            output_dir,
            models_config[api_name],
            num_threads,
            callback
        )

    return model_dirs


def run_recommender_model_pipelines(dataset, output_dir, train_path, test_path, rating_scale, rating_type, models_config, top_k, num_threads, callback):
    model_dirs = []

    for api_name in models_config:
        mp = get_recommender_pipeline_factory()[api_name]()
        model_dirs += mp.run(
            dataset.name,
            train_path,
            test_path,
            rating_scale,
            rating_type,
            output_dir,
            models_config[api_name],
            num_threads,
            callback,
            num_items=top_k
        )

    return model_dirs
