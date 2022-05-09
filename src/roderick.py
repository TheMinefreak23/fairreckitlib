from datetime import datetime

from src.fairreckitlib.data.set.dataset_registry import  *
from fairreckitlib.experiment.parsing.run import Parser
from fairreckitlib.recommender_system import RecommenderSystem

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_from_yml(file_path, rs, num_threads):
    rs.run_experiment_from_yml(file_path, num_threads=num_threads)


def run_dev(file_path, rs, num_threads):
    parser = Parser(True)
    config = parser.parse_experiment_config_from_yml(
        file_path,
        rs.data_registry,
        rs.split_factory,
        rs.model_factory,
        rs.metric_factory
    )

    if config is None:
        print('Failed to parse config:', file_path)
        return

    stamp = str(int(datetime.timestamp(datetime.now())))
    config.name = stamp + '_' + config.name

    rs.run_experiment({}, config, num_threads=num_threads, validate_config=False)


def run_program(rs):
    max_threads = 1
    config_files = [
        #'predictors_all',
        #'predictors_lenskit',
        #'predictors_surprise',
        #'recommenders_all',
        #'recommenders_elliot',
        'recommenders_implicit',
        #'recommenders_lenskit',
        #'recommenders_surprise',
        #'dev_layout',
    ]

    for _, file_name in enumerate(config_files):
        config_path = os.path.join('../config_files', file_name)

        #run_from_yml(config_path, rs, max_threads)
        run_dev(config_path, rs, max_threads)

    #rs.validate_experiment('1650896154_Recommenders_Implicit', 4)


def print_ds(rs, dataset_names=None):
    if dataset_names is None:
        dataset_names = rs.data_registry.get_available()

    for _, dataset_name in enumerate(dataset_names):
        dataset = rs.data_registry.get_set(dataset_name)

        print('')
        print('Dataset', dataset_name)
        print('-------------')
        print('')
        print(dataset_name, 'matrix:')
        print('')
        print(dataset.load_matrix_df())
        for _, table_name in enumerate(dataset.get_available_tables()):
            print('')
            print(dataset_name, 'table:', table_name)
            print('')
            for _, table in enumerate(dataset.read_table(table_name, chunk_size=100000000)):
                print(table)
        print('-------------')


if __name__ == '__main__':
    data_dir_D = os.path.join('D:', 'datasets')
    data_dir_tests = os.path.join('..', 'tests', 'datasets')

    rec_sys = RecommenderSystem(
        data_dir_D,
        os.path.join('..', '..', 'results')
    )

    datasets = [
        #DATASET_LFM_1B,
        #DATASET_LFM_2B,
        DATASET_LFM_360K,
        DATASET_ML_100K,
        DATASET_ML_25M
    ]

    run_program(rec_sys)
    #print_ds(rec_sys, datasets)