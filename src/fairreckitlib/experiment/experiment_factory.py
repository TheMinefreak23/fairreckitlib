"""This module contains functionality to create an experiment factory.

Functions:

    create_experiment_factory: create factory with pipeline factories.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.config.config_factories import GroupFactory
from ..data.data_factory import create_data_factory
from ..data.set.dataset_registry import DataRegistry
from ..evaluation.evaluation_factory import create_evaluation_factory
from ..model.model_factory import create_model_factory


def create_experiment_factory(data_registry: DataRegistry) -> GroupFactory:
    """Create a group factory with all three pipeline factories.

    Consists of three factories:
        1) data factory.
        2) model factory.
        3) evaluation factory.

    Returns:
        the group factory containing the pipeline factories.
    """
    experiment_factory = GroupFactory('experiment')
    experiment_factory.add_factory(create_data_factory(data_registry))
    experiment_factory.add_factory(create_model_factory())
    experiment_factory.add_factory(create_evaluation_factory())
    return experiment_factory
