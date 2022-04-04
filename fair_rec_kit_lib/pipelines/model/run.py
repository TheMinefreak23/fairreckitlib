"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .factory import get_recommender_pipeline_factory


def run_recommender_model_pipelines(dataset, output_dir, train_path, test_path, models_config, top_k, callback):
    model_dirs = []

    for api_name in models_config:
        mp = get_recommender_pipeline_factory()[api_name]()
        model_dirs += mp.run(
            dataset.name,
            train_path,
            test_path,
            output_dir,
            models_config[api_name],
            callback,
            num_items=top_k
        )

    return model_dirs
