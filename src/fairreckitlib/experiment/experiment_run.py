"""This module contains functionality to run the experiment pipelines.

Classes:

    ExperimentPipelineConfig: configuration class to run the experiment pipelines.

Functions:

    run_experiment_pipelines: run the pipeline one or more runs.
    resolve_experiment_start_run: resolve the start run of an existing result directory.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
import os
from typing import Callable, Union

from ..core.config.config_factories import GroupFactory
from ..core.events.event_dispatcher import EventDispatcher
from ..core.io.io_create import create_dir, create_yml
from ..data.set.dataset_registry import DataRegistry
from .experiment_config import PredictorExperimentConfig, RecommenderExperimentConfig
from .experiment_pipeline import ExperimentPipeline


@dataclass
class ExperimentPipelineConfig:
    """Experiment Pipeline Configuration.

    output_dir: the directory to store the output.
    data_registry: the registry with available datasets.
    experiment_factory: the factory with data/model/evaluation pipeline factories.
    experiment_config: the experiment configuration to compute.
    start_run: the experiment run to start with.
    num_runs: the number of runs of the experiment.
    num_threads: the max number of threads the experiment can use.
    """

    output_dir: str
    data_registry: DataRegistry
    experiment_factory: GroupFactory
    experiment_config: Union[PredictorExperimentConfig, RecommenderExperimentConfig]
    start_run: int
    num_runs: int
    num_threads: int


def run_experiment_pipelines(
        pipeline_config: ExperimentPipelineConfig,
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool]) -> bool:
    """Run the experiment pipeline several runs according to the specified pipeline configuration.

    Args:
        pipeline_config: the configuration on how to run the experiment pipelines.
        event_dispatcher: used to dispatch model/IO events when running the experiment pipelines.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Returns:
        whether running the experiment pipelines succeeded.
    """
    if not os.path.isdir(pipeline_config.output_dir):
        # create result output directory
        create_dir(pipeline_config.output_dir, event_dispatcher)

        # save the yml configuration file
        create_yml(
            os.path.join(pipeline_config.output_dir, 'config.yml'),
            pipeline_config.experiment_config.to_yml_format(),
            event_dispatcher
        )

    # prepare pipeline
    experiment_pipeline = ExperimentPipeline(
        pipeline_config.data_registry,
        pipeline_config.experiment_factory,
        event_dispatcher
    )

    start_run = pipeline_config.start_run
    end_run = start_run + pipeline_config.num_runs

    # run the pipeline
    for run in range(start_run, end_run):
        try:
            experiment_pipeline.run(
                os.path.join(pipeline_config.output_dir, 'run_' + str(run)),
                pipeline_config.experiment_config,
                pipeline_config.num_threads,
                is_running
            )
        except RuntimeError:
            return False

    return True


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
