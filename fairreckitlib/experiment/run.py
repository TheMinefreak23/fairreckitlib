"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
import os

import json

from ..data.registry import DataRegistry
from ..data.split.factory import SplitFactory
from ..events import io_event
from ..pipelines.data.run import run_data_pipeline
from ..pipelines.evaluation.run import run_evaluation_pipelines
from ..pipelines.model.factory import ModelFactory
from ..pipelines.model.run import run_model_pipelines
from .constants import EXP_TYPE_RECOMMENDATION


@dataclass
class ExperimentFactories:
    """Experiment Factories wrapper."""

    data_registry: DataRegistry
    split_factory: SplitFactory
    model_factory: ModelFactory


class Experiment:
    """Experiment wrapper of the data, model and evaluation pipelines.

    Args:
        factories(ExperimentFactories): the factories used by the experiment.
        config(ExperimentConfig): the configuration of the experiment.
        event_dispatcher(EventDispatcher): to dispatch the experiment events.
    """
    def __init__(self, factories, config, event_dispatcher):
        self.__factories = factories
        self.__config = config

        self.event_dispatcher = event_dispatcher

    def get_config(self):
        """Gets the configuration of the experiment.

        Returns:
            (ExperimentConfig): the used configuration.
        """
        return self.__config

    def run(self, output_dir, num_threads, is_running):
        """Runs an experiment with the specified configuration.

        Args:
            output_dir(str): the path of the directory to store the output.
            num_threads(int): the max number of threads the experiment can use.
            is_running(func -> bool): function that returns whether the experiment
                is still running. Stops early when False is returned.
        """
        os.mkdir(output_dir)
        self.event_dispatcher.dispatch(
            io_event.ON_MAKE_DIR,
            dir=output_dir
        )

        results = []

        data_result = run_data_pipeline(
            output_dir,
            self.__factories.data_registry,
            self.__factories.split_factory,
            self.__config.datasets,
            self.event_dispatcher,
            is_running
        )

        kwargs = {'num_threads': num_threads}
        if self.__config.type == EXP_TYPE_RECOMMENDATION:
            kwargs['num_items'] = self.__config.top_k

        for data_transition in data_result:
            if not is_running():
                return

            model_dirs = run_model_pipelines(
                data_transition.output_dir,
                data_transition,
                self.__factories.model_factory,
                self.__config.models,
                self.event_dispatcher,
                is_running,
                **kwargs
            )
            if not is_running():
                return

            if len(self.__config.evaluation) > 0:
                run_evaluation_pipelines(
                    data_transition.dataset,
                    data_transition.train_set_path,
                    data_transition.test_set_path,
                    model_dirs,
                    self.__config.evaluation,
                    self.event_dispatcher,
                    **kwargs
                )

            results = add_result_to_overview(results, model_dirs)

        write_storage_file(output_dir, results)


def add_result_to_overview(results, model_dirs):
    """Add result to overview of results file paths"""

    for model_dir in model_dirs:
        # Our evaluations are in the same directory as the model ratings
        result = {
            'dataset': os.path.basename(os.path.dirname(model_dir)),
            'model': os.path.basename(model_dir),
            'dir': model_dir
        }
        results.append(result)

    return results


def write_storage_file(output_dir, results):
    """Write a JSON file with overview of the results file paths"""

    formatted_results = map(lambda result: {
        'name': result['dataset'] + '_' + result['model'],
        'evaluation_path': result['dir'] + '\\evaluations.json',
        'ratings_path': result['dir'] + '\\ratings.tsv',
        'ratings_settings_path': result['dir'] + '\\settings.json'
    }, results)
    output_path = os.path.join(output_dir, 'overview.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump({'overview': list(formatted_results)}, file, indent=4)


def resolve_experiment_start_run(result_dir):
    """Resolves which run will be next in the specified result directory.

    Args:
        result_dir(str): path to the result directory to look into.

    Returns:
        start_run(int): the next run index for this result directory.
    """
    start_run = 0

    for file in os.listdir(result_dir):
        file_name = os.fsdecode(file)
        run_dir = os.path.join(result_dir, file_name)
        if not os.path.isdir(run_dir):
            continue

        run_split = file_name.split('_')
        if len(run_split) != 2:
            continue

        start_run = max(start_run, int(run_split[1]))

    return start_run + 1
