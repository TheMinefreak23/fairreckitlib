"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import time
from dataclasses import dataclass
import os

import json

from ..data.registry import DataRegistry
from ..data.split.factory import SplitFactory
from ..events import io_event, experiment_event
from ..pipelines.data.run import run_data_pipeline
from ..metrics.factory import MetricFactory
from ..pipelines.evaluation.run import run_evaluation_pipelines
from ..pipelines.model.factory import ModelFactory
from ..pipelines.model.run import run_model_pipelines
from .constants import EXP_TYPE_RECOMMENDATION, EXP_KEY_NAME


@dataclass
class ExperimentFactories:
    """Experiment Factories wrapper."""

    data_registry: DataRegistry
    split_factory: SplitFactory
    model_factory: ModelFactory
    metric_factory: MetricFactory


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

        results, start_time = self.start_run(output_dir)

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
            kwargs['rated_items_filter'] = self.__config.rated_items_filter

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
                    model_dirs,
                    data_transition,
                    self.__factories.metric_factory,
                    self.__config.evaluation,
                    self.event_dispatcher,
                    is_running,
                    **kwargs
                )

            results = add_result_to_overview(results, model_dirs)

        self.end_run(start_time, output_dir, results)

    def start_run(self, output_dir):
        """Start the run, making the output dir and initialising the results storage list.

        Args:
            output_dir(str): directory in which to store the run storage output

        Returns:
            results(list): the initial results list
            start_time(float): the time the experiment started
        """
        start_time = time.time()
        self.event_dispatcher.dispatch(
            experiment_event.ON_BEGIN_EXP,
            experiment_name=self.__config.name
        )

        os.mkdir(output_dir)
        self.event_dispatcher.dispatch(
            io_event.ON_MAKE_DIR,
            dir=output_dir
        )

        return [], start_time

    def end_run(self, start_time, output_dir, results):
        """End the run, writing the storage file and storing the results.

        Args:
            start_time(float): time the experiment started
            output_dir(str): directory in which to store the run storage output
            results(list): the results list

        Returns:
        """
        write_storage_file(output_dir, results)

        self.event_dispatcher.dispatch(
            experiment_event.ON_END_EXP,
            experiment_name=self.__config.name,
            elapsed_time=time.time()-start_time
        )


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

