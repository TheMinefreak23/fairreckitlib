""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.experiment.common import EXP_KEY_DATASET_NAME
from .pipeline import DataPipeline


def run_data_pipeline(output_dir, data_registry, split_factory,
                      datasets_config, event_dispatcher):
    """TODO"""
    data_result = []

    data_pipeline = DataPipeline(split_factory, event_dispatcher)
    for data_config in datasets_config:
        dataset_name = data_config[EXP_KEY_DATASET_NAME]
        dataset = data_registry.get_set(dataset_name)

        data_result.append(data_pipeline.run(
            output_dir,
            dataset,
            data_config
        ))

    return data_result
