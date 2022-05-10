"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from dataclasses import dataclass
import os

import json

from ..core.config_constants import TYPE_RECOMMENDATION
from ..core.event_io import ON_MAKE_DIR
from ..data.pipeline.data_run import run_data_pipeline
from ..data.data_factory import KEY_DATASETS
from ..evaluation.pipeline.evaluation_run import run_evaluation_pipelines
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..model.pipeline.model_run import run_model_pipelines
from ..model.model_factory import KEY_MODELS
from .experiment_event import ON_BEGIN_EXPERIMENT, ON_END_EXPERIMENT


class Experiment:
    """Experiment wrapper of the data, model and evaluation pipelines.

    Args:
        data_registry(DataRegistry):
        experiment_factory(GroupFactory):
        config(ExperimentConfig): the configuration of the experiment.
        event_dispatcher(EventDispatcher): to dispatch the experiment events.
    """
    def __init__(self, data_registry, experiment_factory, config, event_dispatcher):
        self.data_registry = data_registry
        self.experiment_factory = experiment_factory
        self.config = config

        self.event_dispatcher = event_dispatcher

    def get_config(self):
        """Gets the configuration of the experiment.

        Returns:
            (ExperimentConfig): the used configuration.
        """
        return self.config

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
            self.data_registry,
            self.experiment_factory.get_factory(KEY_DATASETS),
            self.config.datasets,
            self.event_dispatcher,
            is_running
        )

        kwargs = {'num_threads': num_threads}
        if self.config.type == TYPE_RECOMMENDATION:
            kwargs['num_items'] = self.config.top_k
            kwargs['rated_items_filter'] = self.config.rated_items_filter

        for data_transition in data_result:
            if not is_running():
                return

            model_dirs = run_model_pipelines(
                data_transition.output_dir,
                data_transition,
                self.experiment_factory.get_factory(KEY_MODELS).get_factory(self.config.type),
                self.config.models,
                self.event_dispatcher,
                is_running,
                **kwargs
            )
            if not is_running():
                return

            if len(self.config.evaluation) > 0:
                run_evaluation_pipelines(
                    model_dirs,
                    data_transition,
                    self.experiment_factory.get_factory(KEY_EVALUATION).get_factory(self.config.type),
                    self.config.evaluation,
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
            ON_BEGIN_EXPERIMENT,
            experiment_name=self.config.name
        )

        os.mkdir(output_dir)
        self.event_dispatcher.dispatch(
            ON_MAKE_DIR,
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
            ON_END_EXPERIMENT,
            experiment_name=self.config.name,
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
