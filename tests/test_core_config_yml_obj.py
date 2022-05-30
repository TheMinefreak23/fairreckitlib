"""This module tests the yml and object configuration functionality.

Classes:

    DummyConfig: dummy yml configuration dataclass to use for testing.

Functions:

    test_config_yml: test base yml configuration formatting.
    test_config_object: test configuration object yml formatting.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict

from src.fairreckitlib.core.config.config_object import ObjectConfig
from src.fairreckitlib.core.config.config_yml import YmlConfig, \
    format_yml_config_dict, format_yml_config_list
from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS


@dataclass
class DummyConfig(YmlConfig):
    """Dummy Configuration."""

    value_int: int
    value_float: float

    def to_yml_format(self) -> Dict[str, Any]:
        """Format dummy configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the configuration.
        """
        return {
            'value_int': self.value_int,
            'value_float': self.value_float
        }


def test_config_yml() -> None:
    """Test base yml configuration formatting."""
    yml_config_dict, yml_config_list = {}, []
    for i in range(0, 10):
        dummy_config = DummyConfig(i, float(i))

        yml_config_dict[str(i)] = dummy_config
        yml_config_list.append(dummy_config)

    yml_dict_format = format_yml_config_dict(yml_config_dict)
    assert len(yml_dict_format) == len(yml_config_dict)

    yml_list_format = format_yml_config_list(yml_config_list)
    assert len(yml_list_format) == len(yml_config_list)

    for i in range(0, 10):
        assert yml_dict_format[str(i)] == yml_config_dict[str(i)].to_yml_format()
        assert yml_list_format[i] == yml_config_list[i].to_yml_format()


def test_config_object() -> None:
    """Test configuration object yml formatting."""
    obj_name, obj_params = 'obj', {}
    for i in range(0, 10):
        obj_params['param' + str(i)] = i
    obj = ObjectConfig(obj_name, obj_params)

    yml_format = obj.to_yml_format()
    assert KEY_NAME in yml_format, 'expected name to be included in the yml formatted object.'
    assert KEY_PARAMS in yml_format, 'expected params to be included in the yml formatted object.'

    assert yml_format[KEY_NAME] == obj_name, \
        'expected the formatted object name to be same as the original object name.'
    assert len(yml_format[KEY_PARAMS]) == len(obj_params), \
        'expected all parameters to be present in the formatted yml dictionary.'

    for param_name, param_value in obj_params.items():
        assert param_name in yml_format[KEY_PARAMS], \
            'expected param to be included in the yml formatted object params.'
        assert yml_format[KEY_PARAMS][param_name] == param_value, \
            'expected the formatted param value to be the same as the original param value.'
