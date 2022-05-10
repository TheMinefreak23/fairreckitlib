"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time

from ..core.event_io import ON_MAKE_DIR
from ..core.threading.thread_base import ThreadBase
from .experiment_config import save_config_to_yml
from .experiment_event import ON_BEGIN_THREAD_EXPERIMENT, ON_END_THREAD_EXPERIMENT
from .experiment_run import Experiment


class ThreadExperiment(ThreadBase):
    """Thread that runs the same experiment one or more times."""

    def on_run(self, **kwargs):
        """Runs the experiments.

        Keyword Args:
            output_dir(str): the path of the directory to store the output.
            start_run(int): the initial run index.
            num_runs(int): the number of runs to conduct the experiment.
            factories(ExperimentFactories): the factories used by the experiment.
            config(ExperimentConfig): the configuration of the experiment.
            num_threads(int): the max number of threads the experiment can use.
            on_end_experiment(function): function to execute at the end of the thread experiment.
        """

        output_dir = kwargs['output_dir']
        start_run = kwargs['start_run']
        num_runs = kwargs['num_runs']

        # Create result output directory
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            self.event_dispatcher.dispatch(
                ON_MAKE_DIR,
                dir=output_dir
            )

        save_config_to_yml(os.path.join(output_dir, 'config'), kwargs['config'])

        start_time = time.time()
        self.event_dispatcher.dispatch(
            ON_BEGIN_THREAD_EXPERIMENT,
            num_runs=num_runs,
            experiment_name=kwargs['config'].name
        )

        experiment = Experiment(
            kwargs['registry'],
            kwargs['factory'],
            kwargs['config'],
            self.event_dispatcher
        )

        for run in range(start_run, start_run + num_runs):
            experiment.run(
                os.path.join(output_dir, 'run_' + str(run)),
                kwargs['num_threads'],
                self.is_running
            )

        self.event_dispatcher.dispatch(
            ON_END_THREAD_EXPERIMENT,
            num_runs=num_runs,
            experiment_name=kwargs['config'].name,
            elapsed_time=time.time()-start_time,
        )
