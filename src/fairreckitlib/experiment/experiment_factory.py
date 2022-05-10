"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.factories import GroupFactory
from ..data.data_factory import create_data_factory
from ..evaluation.evaluation_factory import create_evaluation_factory
from ..model.model_factory import create_model_factory


def create_experiment_factory():
    experiment_factory = GroupFactory('experiment')
    experiment_factory.add_factory(create_data_factory())
    experiment_factory.add_factory(create_model_factory())
    experiment_factory.add_factory(create_evaluation_factory())
    return experiment_factory
