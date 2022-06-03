"""This module contains functionality of the complete experiment pipeline.

Classes:

    ExperimentPipeline: class that connects the data, model and evaluation pipelines.

Functions:

    add_result_to_overview: add a computed result to the experiment result overview.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time
from typing import Dict, Callable, List, Tuple, Union

from ..core.config.config_factories import GroupFactory
from ..core.events.event_dispatcher import EventDispatcher
from ..core.io.io_create import create_dir, create_json
from ..data.data_factory import KEY_DATA
from ..data.pipeline.data_run import DataPipelineConfig, run_data_pipelines
from ..data.set.dataset_registry import DataRegistry
from ..evaluation.pipeline.evaluation_run import run_evaluation_pipelines, EvaluationPipelineConfig
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..model.pipeline.model_run import ModelPipelineConfig, run_model_pipelines
from ..model.model_factory import KEY_MODELS
from .experiment_config import ExperimentConfig
from .experiment_config import PredictorExperimentConfig, RecommenderExperimentConfig
from .experiment_event import ON_BEGIN_EXPERIMENT_PIPELINE, ExperimentEventArgs
from .experiment_event import ON_END_EXPERIMENT_PIPELINE


class ExperimentPipeline:
    """ExperimentPipeline that consists of the data, model and evaluation pipelines.

    The experiment pipeline connects the three pipelines, by first running the data
    pipeline for all the specified dataset configurations. Each of the
    resulting data transitions is forwarded through the model pipelines where all the
    specified model configurations will compute rating results. These in turn are
    forwarded to the evaluation pipelines to compute the specified metric configurations
    of the performance of the models.

    Public methods:

    run
    """

    def __init__(
            self,
            data_registry: DataRegistry,
            experiment_factory: GroupFactory,
            event_dispatcher: EventDispatcher):
        """Construct the ExperimentPipeline.

        Args:
            data_registry: the registry with available datasets.
            experiment_factory: the factory containing all three pipeline factories.
            event_dispatcher: to dispatch the experiment events.
        """
        self.data_registry = data_registry
        self.experiment_factory = experiment_factory
        self.event_dispatcher = event_dispatcher

    def run(self,
            output_dir: str,
            experiment_config: Union[PredictorExperimentConfig, RecommenderExperimentConfig],
            num_threads: int,
            is_running: Callable[[], bool]) -> None:
        """Run the experiment with the specified configuration.

        Args:
            output_dir: the path of the directory to store the output.
            experiment_config: the configuration of the experiment.
            num_threads: the max number of threads the experiment can use.
            is_running: function that returns whether the experiment
                is still running. Stops early when False is returned.
        """
        # prepare experiment pipeline
        results, start_time = self.start_run(output_dir, experiment_config)

        # run all data pipelines
        data_result = run_data_pipelines(
            DataPipelineConfig(
                output_dir,
                self.data_registry,
                self.experiment_factory.get_factory(KEY_DATA),
                experiment_config.datasets
            ),
            self.event_dispatcher,
            is_running
        )

        kwargs = {'num_threads': num_threads}
        if isinstance(experiment_config, RecommenderExperimentConfig):
            kwargs['num_items'] = experiment_config.top_k
            kwargs['rated_items_filter'] = experiment_config.rated_items_filter

        # loop through each data transition result from the data pipeline
        for data_transition in data_result:
            if not is_running():
                return

            # run all model pipelines on the data transition
            model_factory = self.experiment_factory.get_factory(KEY_MODELS)
            model_dirs = run_model_pipelines(
                ModelPipelineConfig(
                    data_transition.output_dir,
                    data_transition,
                    model_factory.get_factory(experiment_config.type),
                    experiment_config.models
                ),
                self.event_dispatcher,
                is_running,
                **kwargs
            )
            if not is_running():
                return

            # run all evaluation pipelines on the computed model results
            if len(experiment_config.evaluation) > 0:
                evaluation_factory = self.experiment_factory.get_factory(KEY_EVALUATION)
                run_evaluation_pipelines(
                    EvaluationPipelineConfig(
                        model_dirs,
                        data_transition,
                        evaluation_factory.get_factory(experiment_config.type),
                        experiment_config.evaluation
                    ),
                    self.event_dispatcher,
                    is_running
                )

            # add overview of the data transition on the computed models/metrics
            results = add_result_to_overview(results, model_dirs)

        # finalize experiment pipeline
        self.end_run(start_time, output_dir, experiment_config, results)

    def start_run(
            self,
            output_dir: str,
            experiment_config: ExperimentConfig) -> Tuple[List[Dict[str, str]], float]:
        """Start the run, making the output dir and initializing the results' storage list.

        Args:
            output_dir: directory in which to store the run storage output.
            experiment_config: the configuration of the experiment.

        Returns:
            the initial results list and the time the experiment started.
        """
        start_time = time.time()
        self.event_dispatcher.dispatch(ExperimentEventArgs(
            ON_BEGIN_EXPERIMENT_PIPELINE,
            experiment_config.name
        ))

        create_dir(output_dir, self.event_dispatcher)

        return [], start_time

    def end_run(
            self,
            start_time: float,
            output_dir: str,
            experiment_config: ExperimentConfig,
            results: List[Dict[str, str]]) -> None:
        """End the run, writing the storage file and storing the results.

        Args:
            start_time: time the experiment started.
            output_dir: directory in which to store the run storage output.
            experiment_config: the configuration of the experiment.
            results: the current results list.
        """
        self.write_storage_file(output_dir, results)

        self.event_dispatcher.dispatch(ExperimentEventArgs(
            ON_END_EXPERIMENT_PIPELINE,
            experiment_config.name
        ), elapsed_time=time.time() - start_time)

    def write_storage_file(
            self,
            output_dir: str,
            results: List[Dict[str, str]]) -> None:
        """Write a JSON file with overview of the results file paths.

        Args:
            output_dir: path to the directory to store the result overview.
            results: the result overview containing completed computations.
        """
        formatted_results = map(lambda result: {
            'name': result['dataset'] + ' - ' + result['model'],
            'dataset': result['dataset'],
            'recommender_system': result['model'],
            'evaluation_path': result['dir'] + '\\evaluations.json',
            'ratings_path': result['dir'] + '\\ratings.tsv',
            'ratings_settings_path': result['dir'] + '\\settings.json'
        }, results)

        create_json(
            os.path.join(output_dir, 'overview.json'),
            {'overview': list(formatted_results)},
            self.event_dispatcher,
            indent=4
        )


def add_result_to_overview(
        results: List[Dict[str, str]],
        model_dirs: List[str]) -> List[Dict[str, str]]:
    """Add result to overview of results file paths.

    Args:
        results: the accumulated result overview.
        model_dirs: the completed computations to add to the overview.

    Returns:
        the result overview appended with the completed computations.
    """
    for model_dir in model_dirs:
        # Our evaluations are in the same directory as the model ratings
        result = {
            'dataset': os.path.basename(os.path.dirname(model_dir)),
            'model': os.path.basename(model_dir),
            'dir': model_dir
        }
        results.append(result)

    return results
