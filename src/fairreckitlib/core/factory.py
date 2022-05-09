"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta

from .params import get_empty_parameters

FUNC_CREATE_OBJ = 'f_create_obj'
FUNC_CREATE_PARAMS = 'f_create_params'


class BaseFactory(metaclass=ABCMeta):

    def __init__(self, factory_name):
        self.factory_name = factory_name
        self.factory = {}

    def get_available_names(self):
        result = []

        for name, _ in self.factory.items():
            result.append(name)

        return result

    def get_name(self):
        return self.factory_name


class Factory(BaseFactory):

    def add(self, obj_name, func_create_obj, func_create_obj_params=None):
        if obj_name in self.factory:
            raise KeyError('Factory ' + self.factory_name,
                           ': obj already exists: ' + obj_name)

        if func_create_obj_params is None:
            func_create_obj_params = get_empty_parameters

        self.factory[obj_name] = {
            FUNC_CREATE_OBJ: func_create_obj,
            FUNC_CREATE_PARAMS: func_create_obj_params
        }

    def create(self, obj_name, obj_params=None, **kwargs):
        if obj_name not in self.factory:
            return None

        func_create_obj = self.factory[obj_name][FUNC_CREATE_OBJ]
        if obj_params is None:
            obj_params = self.create_params(obj_name)

        return func_create_obj(obj_name, dict(obj_params), **kwargs)

    def create_params(self, obj_name):
        if obj_name not in self.factory:
            return get_empty_parameters()

        return self.factory[obj_name][FUNC_CREATE_PARAMS]()

    def get_available(self):
        obj_list = []

        for obj_name, entry in self.factory.items():
            obj_params = entry[FUNC_CREATE_PARAMS]()
            obj_list.append({
                'name': obj_name,
                'params': obj_params.to_dict()
            })

        return obj_list


class GroupFactory(BaseFactory):

    def add_factory(self, factory):
        factory_name = factory.get_name()
        if factory_name in self.factory:
            raise KeyError()

        self.factory[factory_name] = factory

    def create(self, factory_name, obj_name, obj_params=None, **kwargs):
        if factory_name not in self.factory:
            return None

        return self.factory[factory_name].create(obj_name, obj_params, **kwargs)

    def create_params(self, factory_name, obj_name):
        if factory_name not in self.factory:
            return get_empty_parameters()

        return self.factory[factory_name].create_params(obj_name)

    def get_factory(self, factory_name):
        return self.factory.get(factory_name)


def create_factory_from_list(factory_name, obj_tuple_list):
    factory = Factory(factory_name)

    for _, obj in enumerate(obj_tuple_list):
        (obj_name, func_create_obj, func_create_obj_params) = obj
        factory.add(
            obj_name,
            func_create_obj,
            func_create_obj_params
        )

    return factory
