"""This module tests the creation of factories.

Functions:

    create_dummy_obj: create dummy that returns function arguments.
    create_dummy_params: create dummy config with parameter(s).
    test_factory_add_and_available: test adding and checking objects in factory.
    test_factory_create: test object creation with params and kwargs.
    test_factory_create_params: test parameter creation.
    test_factory_create_from_tuples: test creation from tuple list.
    test_factory_name: check if names are correct.
    test_group_factory_add_and_available: test adding and checking objects in group factory.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pytest

from src.fairreckitlib.core.config_constants import KEY_NAME, KEY_PARAMS
from src.fairreckitlib.core.config_params import ConfigParameters, create_empty_parameters
from src.fairreckitlib.core.factories import Factory, GroupFactory, create_factory_from_list

dummy_names = ['dummy_a', 'dummy_b', 'dummy_c']


def create_dummy_obj(name, params, **kwargs):
    """Create dummy object that returns the function arguments."""
    return name, params, kwargs


def create_dummy_params():
    """Create dummy config that has at least one parameter."""
    params = ConfigParameters()
    params.add_random_seed('seed')
    return params


def test_factory_add_and_available():
    """Test adding objects to factory and (name) availability."""
    factory = Factory('factory')
    for i, name in enumerate(dummy_names):
        assert not factory.is_obj_available(name), 'object should not be available'
        assert factory.get_num_entries() == i

        factory.add_obj(name, create_dummy_obj, None)

        assert factory.get_num_entries() == i + 1
        assert factory.is_obj_available(name), 'object should be available'

        pytest.raises(KeyError, factory.add_obj, name, create_dummy_obj, None)

        assert name in factory.get_available_names(), 'object should be available'

    availability = factory.get_available()
    assert len(availability) == factory.get_num_entries(), 'availability should have the same' \
                                                           'length as the number of entries'

    for _, entry in enumerate(availability):
        assert KEY_NAME in entry, 'each entry should have a name'
        assert KEY_PARAMS in entry, 'each entry should have parameters'
        assert entry[KEY_NAME] in dummy_names, 'name of the entry should be in the original list.'


def test_factory_create():
    """Test object creation with parameters and keyword arguments."""
    obj_name = 'obj'
    obj_kwargs = { 'kwargs': True }
    obj_params = { 'params': True }

    factory = Factory('factory')
    factory.add_obj(obj_name, create_dummy_obj, create_dummy_params)

    assert factory.create(obj_name + '1') is None, 'object should not exist.'

    # check creation with default params and no kwargs
    name, params, kwargs = factory.create(obj_name, None, **{})

    assert len(params) == create_dummy_params().get_num_params(), 'expected default params'
    assert len(kwargs) == 0, 'expected no keyword arguments'

    # check creation with params and kwargs
    name, params, kwargs = factory.create(obj_name, obj_params, **obj_kwargs)

    assert name == obj_name, 'expected object name.'
    assert params['params'], 'expected object params.'
    assert kwargs['kwargs'], 'expected object kwargs.'


@pytest.mark.parametrize('create_params', [None, create_dummy_params])
def test_factory_create_params(create_params):
    """Test parameter creation for objects."""
    factory = Factory('factory')
    factory.add_obj('obj', create_dummy_obj, create_params)

    params = factory.create_params('obj')
    assert params.get_num_params() == 0 if create_params is None else 1

    params = factory.create_params('obj1')
    assert params.get_num_params() == 0, 'non existing object should have no parameters.'


def test_factory_create_from_tuples():
    """Test factory creation from a list of object name/create/parameters tuples."""
    obj_tuple_list = []

    factory = create_factory_from_list('factory', obj_tuple_list)
    assert factory.get_num_entries() == 0, 'factory should have no entries for an empty list.'

    obj_tuple_list = [(name, create_dummy_obj, create_empty_parameters) for name in dummy_names]

    factory = create_factory_from_list('factory', obj_tuple_list)
    assert factory.get_num_entries() == len(obj_tuple_list), 'factory should have all tuples' \
                                                             'added after creation.'


@pytest.mark.parametrize('create_factory', [Factory, GroupFactory])
def test_factory_name(create_factory):
    """Test (group) factory names are correct."""
    for _, name in enumerate(dummy_names):
        factory = create_factory(name)
        assert factory.get_name() == name, 'factory name should be the same after creation.'


@pytest.mark.parametrize('create_child_factory', [Factory, GroupFactory])
def test_group_factory_add_and_available(create_child_factory):
    """Test adding child factories to group factory and (name) availability."""
    group = GroupFactory('group')
    for i, name in enumerate(dummy_names):
        factory = create_child_factory(name)

        assert group.get_factory(name) is None, 'factory should not be available.'
        assert group.get_num_entries() == i

        group.add_factory(factory)

        assert group.get_factory(name) is not None, 'factory should be available.'
        assert group.get_num_entries() == i + 1

        pytest.raises(KeyError, group.add_factory, factory)

        assert name in group.get_available_names(), 'factory should be in available names.'
        assert not group.is_obj_available(name), 'factory should not be available, only objects.'

    availability = group.get_available()
    assert len(availability) == group.get_num_entries(), 'availability should have the same' \
                                                         ' length as the number of entries'

    child_name = 'child'
    child = Factory('child')
    child.add_obj('obj', create_dummy_obj, None)
    group.add_factory(child)
    assert group.is_obj_available('obj'), 'obj should be available in the child factory.'

    for factory_name, factory_availability in availability.items():
        if factory_name == child_name:
            assert len(factory_availability) == 1, 'child factory should have one available entry.'
        else:
            assert factory_name in dummy_names, 'factory name should be in the original list'
            assert len(factory_availability) == 0, 'each original factory has no availability'
