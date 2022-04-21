"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os.path
from dataclasses import dataclass

from ..data.registry import DataRegistry
from ..data.split.factory import SplitFactory
from ..events.data_event import get_data_events
from ..events.io_event import get_io_events
from ..events.model_event import get_model_events
from ..events.evaluation_event import get_evaluation_events
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
        event_dispatcher(EventDispatcher): to dispatch the experiment events.
        verbose(bool): whether to display extended information in the console.
    """
    def __init__(self, factories, event_dispatcher, verbose=True):
        self.__factories = factories

        self.verbose = verbose
        self.event_dispatcher = event_dispatcher

    def run(self, output_dir, config, num_threads):
        """Runs an experiment with the specified configuration.

        Args:
            output_dir(str): the path of the directory to store the output.
            config(ExperimentConfig): the configuration of the experiment.
            num_threads(int): the max number of threads the model pipeline can use.
        """
        self.__attach_event_listeners()

        results = []

        data_result = run_data_pipeline(
            output_dir,
            self.__factories.data_registry,
            self.__factories.split_factory,
            config.datasets,
            self.event_dispatcher
        )

        for data_transition in data_result:

            kwargs = {'num_threads': num_threads}
            if config.type == EXP_TYPE_RECOMMENDATION:
                kwargs['num_items'] = config.top_k

            model_dirs = run_model_pipelines(
                data_transition.output_dir,
                data_transition,
                self.__factories.model_factory,
                config.models,
                self.event_dispatcher,
                **kwargs
            )

            # TODO temp workaround for empty evaluation config
            if len(config.evaluation) > 0:
                run_evaluation_pipelines(
                    data_transition.dataset,
                    data_transition.train_set_path,
                    data_transition.test_set_path,
                    model_dirs,
                    config.evaluation,
                    self.event_dispatcher,
                    **kwargs
                )

            results = add_result_to_overview(results, model_dirs)

        self.__detach_event_listeners()

        return results

    def __attach_event_listeners(self):
        event_listeners = Experiment.get_events()
        for _, (event_id, func_on_event) in enumerate(event_listeners):
            self.event_dispatcher.add_listener(event_id, self, func_on_event)

    def __detach_event_listeners(self):
        event_listeners = Experiment.get_events()
        for _, (event_id, func_on_event) in enumerate(event_listeners):
            self.event_dispatcher.remove_listener(event_id, self, func_on_event)

    @staticmethod
    def get_events():
        """Gets all experiment events.

        Returns:
            (array like) list of pairs in the format (event_id, func_on_event)
        """
        events = get_io_events()

        events += get_data_events()
        events += get_model_events()
        events += get_evaluation_events()

        return events


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

def run_experiment(output_dir, experiment_factories, experiment_config,
                   event_dispatcher, num_threads=0, verbose=True):
    """Runs an experiment with the specified configuration.

    Args:
        output_dir(str): the path of the directory to store the output.
        experiment_factories(ExperimentFactories): the factories used by the experiment.
        experiment_config(ExperimentConfig): the configuration of the experiment.
        event_dispatcher(EventDispatcher): to dispatch the experiment events.
        num_threads(int): the max number of threads the model pipeline can use.
        verbose(bool): whether to display extended information in the console.
    """
    experiment = Experiment(
        experiment_factories,
        event_dispatcher,
        verbose=verbose
    )

    return experiment.run(
        output_dir,
        experiment_config,
        num_threads
    )
