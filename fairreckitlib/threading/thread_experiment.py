"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from fairreckitlib.experiment.run import Experiment
from .thread_base import ThreadBase


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
        """
        output_dir = kwargs['output_dir']
        start_run = kwargs['start_run']
        num_runs = kwargs['num_runs']

        experiment = Experiment(
            kwargs['factories'],
            kwargs['config'],
            self.event_dispatcher
        )

        for run in range(start_run, start_run + num_runs):
            experiment.run(
                os.path.join(output_dir, 'run_' + str(run)),
                kwargs['num_threads'],
                self.is_running
            )
