import os

from fairreckitlib.data.set import DATASET_ML_100K
from fairreckitlib.data.split.factory import SPLIT_RANDOM
from fairreckitlib.experiment.common import *
from fairreckitlib.experiment.config import create_config_dataset, create_config_all_recommender_models
from fairreckitlib.recommender_system import RecommenderSystem


def create_recommender_experiment_config(experiment_name):
    datasets = [
        create_config_dataset(DATASET_ML_100K, 0.2, SPLIT_RANDOM),
        # create_config_dataset(DATASET_ML_100K, 0.3, SPLIT_RANDOM)
    ]
    models = create_config_all_recommender_models(
        elliot=False,
        implicit=True,
        lenskit=False
    )

    from fairreckitlib.metrics.metrics2 import Metric
    from fairreckitlib.metrics.filter import Filter
    metrics = [Metric.precision, Metric.recall, Metric.mrr, Metric.rmse, Metric.item_coverage, Metric.user_coverage]
    gender_filter = {'type': Filter.Equals.value, 'name': 'gender', 'value': 'male'}
    country_filter = {'type': Filter.Equals.value, 'name': 'country', 'value': 'Mexico'}
    age_filter = {'type': Filter.Clamp.value, 'name': 'age', 'value': {'min': 15, 'max': 25}}
    filters = [
        [gender_filter],
        [country_filter, age_filter]  # Multiple filter 'passes'
    ]
    evaluation = {
        'metrics': metrics,
        'filters': filters
    }

    return {
        EXP_KEY_NAME: experiment_name,
        EXP_KEY_TYPE: EXP_TYPE_RECOMMENDATION,
        EXP_KEY_TOP_K: 10,
        EXP_KEY_DATASETS: datasets,
        EXP_KEY_MODELS: models,
        EXP_KEY_EVALUATION: evaluation
    }


def run_recommender_experiment(recommender_system, num_threads):
    from datetime import datetime
    stamp = str(int(datetime.timestamp(datetime.now())))
    config = create_recommender_experiment_config(stamp + '_HelloFRK')

    """
    with open(os.path.join('..', 'recommenders.yml'), 'w') as file:
        yaml.dump(config, file)
    """

    recommender_system.run_experiment(config, num_threads=num_threads)

if __name__ == '__main__':
    rs = RecommenderSystem(
        os.path.join('..', 'datasets'),
        os.path.join('..', 'results')
    )

    # print_available_datasets(rs)
    # print_available_predictors(rs)
    # print_available_recommenders(rs)

    max_threads = 1

    #run_prediction_experiment(rs, max_threads)
    run_recommender_experiment(rs, max_threads)