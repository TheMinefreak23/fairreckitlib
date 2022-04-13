"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..events.data_event import get_data_events
from ..events.model_event import get_model_events
from ..events.evaluation_event import get_evaluation_events
from ..pipelines.data.run import run_data_pipeline
from ..pipelines.evaluation.run import run_evaluation_pipelines
from ..pipelines.model.run import run_model_pipelines
from .common import EXP_KEY_DATASETS, EXP_KEY_MODELS, EXP_KEY_EVALUATION
from .common import EXP_KEY_TOP_K
from .common import EXP_KEY_TYPE
from .common import EXP_TYPE_RECOMMENDATION


class Experiment:

    def __init__(self, data_registry, split_factory, model_factory,
                 event_dispatcher, verbose=True):
        self.__data_registry = data_registry
        self.__split_factory = split_factory
        self.__model_factory = model_factory

        self.verbose = verbose
        self.event_dispatcher = event_dispatcher

    def run(self, output_dir, config, num_threads):
        self.__attach_event_listeners()

        data_result = run_data_pipeline(
            output_dir,
            self.__data_registry,
            self.__split_factory,
            config[EXP_KEY_DATASETS],
            self.event_dispatcher
        )

        for data_transition in data_result:
            kwargs = {'num_threads': num_threads}
            if config[EXP_KEY_TYPE] is EXP_TYPE_RECOMMENDATION:
                kwargs['num_items'] = config[EXP_KEY_TOP_K]

            model_dirs = run_model_pipelines(
                data_transition.output_dir,
                data_transition,
                self.__model_factory,
                config[EXP_KEY_MODELS],
                self.event_dispatcher,
                **kwargs
            )

            run_evaluation_pipelines(
                data_transition.dataset,
                data_transition.train_set_path,
                data_transition.test_set_path,
                model_dirs,
                config[EXP_KEY_EVALUATION],
                self.event_dispatcher,
                **kwargs
            )

        self.__detach_event_listeners()

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
        events = []
        events += get_data_events()
        events += get_model_events()
        events += get_evaluation_events()
        return events
