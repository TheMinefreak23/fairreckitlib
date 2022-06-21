"""This module contains the parameter creation functions for filters.

Functions:

    create_params_numerical: TODO
    create_params_categorical: TODO
    create_params_count: TODO

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.config.config_parameters import ConfigParameters


def create_params_numerical(**kwargs) -> ConfigParameters:
    """Create the parameters of a numerical filter.

    Keyword Args:
        column_name(str): the name of the column that has numerical values.
        dataset(Dataset): the dataset associated with the filter.
        matrix_name(str): the matrix name related to the dataset.

    Returns:
        the configuration parameters of the numerical filter.
    """
    column_name = kwargs['column_name']
    dataset = kwargs['dataset']
    matrix_name = kwargs['matrix_name']

    params = ConfigParameters()

    for table_name, table_columns in dataset.get_available_columns(matrix_name).items():
        if column_name in table_columns:
            table = dataset.read_table(table_name, [column_name])
            numerical_range = (int(table[column_name].min()), int(table[column_name].max()))
            params.add_range('range', int, numerical_range, numerical_range)

    return params


def create_params_categorical(**kwargs) -> ConfigParameters:
    """Create the parameters of a categorical filter.

    Keyword Args:
        column_name(str): the name of the column that has categorical values.
        dataset(Dataset): the dataset associated with the filter.
        matrix_name(str): the matrix name related to the dataset.

    Returns:
        the configuration parameters of the categorical filter.
    """
    column_name = kwargs['column_name']
    dataset = kwargs['dataset']
    matrix_name = kwargs['matrix_name']

    params = ConfigParameters()

    for table_name, table_columns in dataset.get_available_columns(matrix_name).items():
        if column_name in table_columns:
            table = dataset.read_table(table_name, [column_name])
            categories = list(table[column_name].fillna('').unique())
            categories = [None if len(c) == 0 else c for c in categories]
            params.add_multi_option('values', categories, categories)

    return params

def create_params_count(**_) -> ConfigParameters:
    """Create the parameters of a count filter.

    Returns:
        the configuration parameters of the count filter.
    """
    params = ConfigParameters()
    params.add_number('threshold', int, 100, (1, 10000000000))
    return params
