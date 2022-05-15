"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time
from typing import Dict, Callable, List, Tuple, Union

import json

from ..core.event_dispatcher import EventDispatcher
from ..core.event_io import ON_MAKE_DIR
from ..core.factories import GroupFactory
from ..data.data_factory import KEY_DATASETS
from ..data.pipeline.data_run import run_data_pipeline
from ..data.set.dataset_registry import DataRegistry
from ..evaluation.pipeline.evaluation_run import run_evaluation_pipelines
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..model.pipeline.model_run import run_model_pipelines
from ..model.model_factory import KEY_MODELS
from .experiment_config import PredictorExperimentConfig, RecommenderExperimentConfig
from .experiment_event import ON_BEGIN_EXPERIMENT, ON_END_EXPERIMENT


class Experiment:
    """Experiment wrapper of the data, model and evaluation pipelines."""

    def __init__(
            self,
            data_registry: DataRegistry,
            experiment_factory: GroupFactory,
            experiment_config: Union[PredictorExperimentConfig, RecommenderExperimentConfig],
            event_dispatcher: EventDispatcher):
        """Construct the experiment.

        Args:
            data_registry(DataRegistry): the registry with available datasets.
            experiment_factory(GroupFactory): the factory containing all three pipeline factories.
            experiment_config: the configuration of the experiment.
            event_dispatcher: to dispatch the experiment events.
        """
        self.data_registry = data_registry
        self.experiment_factory = experiment_factory
        self.experiment_config = experiment_config
        self.event_dispatcher = event_dispatcher

    def get_config(self) -> Union[PredictorExperimentConfig, RecommenderExperimentConfig]:
        """Get the configuration of the experiment.

        Returns:
            the used configuration.
        """
        return self.experiment_config

    def run(self, output_dir: str, num_threads: int, is_running: Callable[[], bool]) -> None:
        """Run the experiment with the specified configuration.

        Args:
            output_dir: the path of the directory to store the output.
            num_threads: the max number of threads the experiment can use.
            is_running: function that returns whether the experiment
                is still running. Stops early when False is returned.
        """
        results, start_time = self.start_run(output_dir)

        data_result = run_data_pipeline(
            output_dir,
            self.data_registry,
            self.experiment_factory.get_factory(KEY_DATASETS),
            self.experiment_config.datasets,
            self.event_dispatcher,
            is_running
        )

        kwargs = {'num_threads': num_threads}
        if isinstance(self.experiment_config, RecommenderExperimentConfig):
            kwargs['num_items'] = self.experiment_config.top_k
            kwargs['rated_items_filter'] = self.experiment_config.rated_items_filter

        for data_transition in data_result:
            if not is_running():
                return

            model_factory = self.experiment_factory.get_factory(KEY_MODELS)
            model_dirs = run_model_pipelines(
                data_transition.output_dir,
                data_transition,
                model_factory.get_factory(self.experiment_config.type),
                self.experiment_config.models,
                self.event_dispatcher,
                is_running,
                **kwargs
            )
            if not is_running():
                return

            if len(self.experiment_config.evaluation) > 0:
                evaluation_factory = self.experiment_factory.get_factory(KEY_EVALUATION)
                run_evaluation_pipelines(
                    model_dirs,
                    data_transition,
                    evaluation_factory.get_factory(self.experiment_config.type),
                    self.experiment_config.evaluation,
                    self.event_dispatcher,
                    is_running,
                    **kwargs
                )

            results = add_result_to_overview(results, model_dirs)

        self.end_run(start_time, output_dir, results)

    def start_run(self, output_dir: str) -> Tuple[List[Dict[str, str]], float]:
        """Start the run, making the output dir and initialising the results' storage list.

        Args:
            output_dir: directory in which to store the run storage output.

        Returns:
            the initial results list and the time the experiment started.
        """
        start_time = time.time()
        self.event_dispatcher.dispatch(
            ON_BEGIN_EXPERIMENT,
            experiment_name=self.experiment_config.name
        )

        os.mkdir(output_dir)
        self.event_dispatcher.dispatch(
            ON_MAKE_DIR,
            dir=output_dir
        )

        return [], start_time

    def end_run(self, start_time: float, output_dir: str, results: List[Dict[str, str]]) -> None:
        """End the run, writing the storage file and storing the results.

        Args:
            start_time: time the experiment started.
            output_dir: directory in which to store the run storage output.
            results: the current results list.
        """
        write_storage_file(output_dir, results)

        self.event_dispatcher.dispatch(
            ON_END_EXPERIMENT,
            experiment_name=self.experiment_config.name,
            elapsed_time=time.time()-start_time
        )


def add_result_to_overview(
        results: List[Dict[str, str]],
        model_dirs: List[str]) -> List[Dict[str, str]]:
    """Add result to overview of results file paths."""
    for model_dir in model_dirs:
        # Our evaluations are in the same directory as the model ratings
        result = {
            'dataset': os.path.basename(os.path.dirname(model_dir)),
            'model': os.path.basename(model_dir),
            'dir': model_dir
        }
        results.append(result)

    return results


def write_storage_file(output_dir: str, results: List[Dict[str, str]]) -> None:
    """Write a JSON file with overview of the results file paths."""
    formatted_results = map(lambda result: {
        'name': result['dataset'] + '_' + result['model'],
        'evaluation_path': result['dir'] + '\\evaluations.json',
        'ratings_path': result['dir'] + '\\ratings.tsv',
        'ratings_settings_path': result['dir'] + '\\settings.json'
    }, results)
    output_path = os.path.join(output_dir, 'overview.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump({'overview': list(formatted_results)}, file, indent=4)


def resolve_experiment_start_run(result_dir: str) -> int:
    """Resolve which run will be next in the specified result directory.

    Args:
        result_dir: path to the result directory to look into.

    Returns:
        the next run index for this result directory.
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
