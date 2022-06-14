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
    format_yml_config_dict, format_yml_config_dict_list, format_yml_config_list
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
    """Test base yml configuration formatting with containers."""
    yml_config_dict, yml_config_dict_list, yml_config_list = {}, {}, []
    for i in range(0, 10):
        dummy_config = DummyConfig(i, float(i))
        key_name = str(i)

        yml_config_dict[key_name] = dummy_config
        yml_config_list.append(dummy_config)
        yml_config_dict_list[key_name] = list(yml_config_list)

    # test dict with yml config
    yml_dict_format = format_yml_config_dict(yml_config_dict)
    assert len(yml_dict_format) == len(yml_config_dict), \
        'expected the format dictionary to be of the same length as the config dictionary'
    for key_name, _ in yml_dict_format.items():
        assert key_name in yml_config_dict, \
            'expected key in the format dictionary to be present in the config dictionary'

    # test dict with list of yml configs
    yml_dict_list_format = format_yml_config_dict_list(yml_config_dict_list)
    assert len(yml_dict_list_format) == len(yml_config_dict_list), \
        'expected the format dictionary to be of the same length as the config dictionary'
    for key_name, yml_list_format in yml_dict_list_format.items():
        assert key_name in yml_config_dict_list, \
            'expected key in the format dictionary to be present in the config dictionary'
        assert len(yml_list_format) == len(yml_config_dict_list[key_name]), \
            'expected format list to be the same length as the config list'

    # test list of yml configs
    yml_list_format = format_yml_config_list(yml_config_list)
    assert len(yml_list_format) == len(yml_config_list), \
        'expected the format list to be of the same length as the config list'

    # test values of all three yml containers
    for i in range(0, 10):
        key_name = str(i)
        assert yml_dict_format[key_name] == yml_config_dict[key_name].to_yml_format(), \
            'expected format dictionary entry to be the same as in the config dictionary'
        assert yml_list_format[i] == yml_config_list[i].to_yml_format(), \
            'expected format list entry to be the same as in the config list'
        for j, yml_config in enumerate(yml_dict_list_format[key_name]):
            assert yml_config == yml_config_dict_list[key_name][j].to_yml_format(), \
                'expected format dictionary list entry to be the same as in the config list'


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
