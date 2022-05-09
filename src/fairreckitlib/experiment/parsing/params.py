"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.events import config_event
from src.fairreckitlib.experiment.parsing import assertion


def parse_config_parameters(params_config, parent_name, parameters, event_dispatcher):
    """Parses the model parameters' configuration.

    Args:
        params_config(dict): dictionary with the configuration to parse.
        parent_name(str): name of the parent related to the parameters' configuration.
        parameters(ConfigParameters): the configuration parameters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_params(dict): the parsed params configuration as key-value pairs.
    """
    # start with parameter defaults
    parsed_params = parameters.get_defaults()

    # assert params_config is a dict
    if not assertion.is_type(
        params_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: ' + parent_name + ' invalid params value',
        default=parsed_params
    ): return parsed_params

    # remove unnecessary parameters from configuration
    params_config = trim_config_params(
        params_config,
        parent_name,
        parameters,
        event_dispatcher
    )

    # assert params_config has entries left after trimming
    if not assertion.is_container_not_empty(
        params_config,
        event_dispatcher,
        'PARSE WARNING: ' + parent_name + ' params is empty',
        default=parsed_params
    ): return parsed_params

    # parse params_config entries
    for param_name, _ in parsed_params.items():
        success, value = parse_config_param(
            params_config,
            parent_name,
            parameters.get_param(param_name),
            event_dispatcher
        )

        # replace defaults on success
        if success:
            parsed_params[param_name] = value

    return parsed_params


def parse_config_param(params_config, parent_name, param, event_dispatcher):
    """Parses a parameter from the specified configuration.

    Args:
        params_config(dict): dictionary with the parameters' configuration.
        parent_name(str): name of the parent related to the parameters' configuration.
        param(ConfigParam): the parameter that is being parsed.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        success(bool): whether the parsing succeeded.
        value: the parsed and validated value.
    """
    param_default = param.default_value

    # assert param_name is present in the configuration
    if not assertion.is_key_in_dict(
        param.name,
        params_config,
        event_dispatcher,
        'PARSE WARNING: ' + parent_name + ' missing param for \'' + param.name + '\'',
        default=param_default
    ): return False, param_default

    config_value = params_config[param.name]
    # validate the configuration value
    success, value, error_msg = param.validate_value(config_value)

    if not success:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg='PARSE WARNING: ' + parent_name + ' invalid param \'' + param.name + '\'' +
                '\n\t' + error_msg,
            actual=config_value,
            default=value
        )
    # validation succeeded but extra info is available
    elif len(error_msg) > 0:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg='PARSE WARNING: ' + parent_name + ' modified param \'' + param.name + '\'' +
                '\n\t' + error_msg,
            actual=config_value
        )

    return success, value


def trim_config_params(params_config, parent_name, parameters, event_dispatcher):
    """Trims parameters from the specified configuration.

    Removes unnecessary parameters that are not present in the
    original config parameter list.

    Args:
        params_config(dict): dictionary with the parameters' configuration.
        parent_name(str): name of the parent related to the parameters' configuration.
        parameters(ConfigParameters): the configuration parameters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        trimmed_config(dict): dictionary with the trimmed parameters.
    """
    trimmed_config = {}

    for param_name, param_value in params_config.items():
        if assertion.is_one_of_list(
            param_name,
            parameters.get_param_names(),
            event_dispatcher,
            'PARSE WARNING: ' + parent_name + ' unknown parameter \'' + param_name + '\''
        ): trimmed_config[param_name] = param_value

    return trimmed_config
