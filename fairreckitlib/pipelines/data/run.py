""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.data.split.factory import get_split_factory
from fairreckitlib.experiment.config import EXP_KEY_DATASET_NAME
from fairreckitlib.experiment.config import EXP_KEY_DATASET_PREFILTERS
from fairreckitlib.experiment.config import EXP_KEY_DATASET_RATING_MODIFIER
from fairreckitlib.experiment.config import EXP_KEY_DATASET_SPLIT
from .pipeline import DataPipeline

def run_data_pipeline(output_dir, data_registry, datasets_config, callback):
    data_tuples = []

    dp = DataPipeline(get_split_factory())
    for config in datasets_config:
        dataset_name = config[EXP_KEY_DATASET_NAME]
        dataset = data_registry.get_set(dataset_name)

        data_dir, train_set_path, test_set_path = dp.run(
            output_dir,
            dataset,
            config[EXP_KEY_DATASET_PREFILTERS],
            config[EXP_KEY_DATASET_RATING_MODIFIER],
            config[EXP_KEY_DATASET_SPLIT],
            callback
        )

        tuple = (dataset, data_dir, train_set_path, test_set_path)
        data_tuples.append(tuple)

    return data_tuples
