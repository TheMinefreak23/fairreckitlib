"""This module contains a parser for the dataset configuration.

Functions:

    parse_dataset_file_config: parse dataset file configuration.
    parse_dataset_table_config: parse dataset table configuration.
    parse_dataset_index_config: parse dataset matrix' user/item configuration.
    parse_dataset_matrix_config: parse dataset matrix configuration.
    parse_dataset_config: parse dataset configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Optional

from ...core.config_constants import KEY_NAME
from .dataset_constants import KEY_DATASET, KEY_EVENTS, KEY_MATRICES, KEY_TABLES
from .dataset_constants import KEY_MATRIX, KEY_IDX_ITEM, KEY_IDX_USER
from .dataset_constants import KEY_RATING_MIN, KEY_RATING_MAX, KEY_RATING_TYPE
from .dataset_constants import TABLE_KEY, TABLE_PRIMARY_KEY, TABLE_FOREIGN_KEYS, TABLE_COLUMNS
from .dataset_constants import TABLE_FILE, TABLE_COMPRESSION, TABLE_ENCODING
from .dataset_constants import TABLE_HEADER, TABLE_INDEXED, TABLE_NUM_RECORDS, TABLE_SEP
from .dataset_config import DatasetConfig, DatasetIndexConfig, DatasetMatrixConfig
from .dataset_config import DatasetFileConfig, DatasetTableConfig


def parse_dataset_file_config(
        file_config: Dict[str, Any]) -> Optional[DatasetFileConfig]:
    """Parse a dataset file configuration.

    Args:
        file_config: the dataset file configuration.

    Returns:
        the parsed configuration or None on failure.
    """
    # TODO parse this
    return DatasetFileConfig(
        file_config[KEY_NAME],
        file_config.get(TABLE_SEP),
        file_config.get(TABLE_COMPRESSION),
        file_config.get(TABLE_ENCODING),
        file_config.get(TABLE_HEADER, False),
        file_config.get(TABLE_INDEXED, False)
    )


def parse_dataset_table_config(
        table_config: Dict[str, Any]) -> Optional[DatasetTableConfig]:
    """Parse a dataset table configuration.

    Args:
        table_config: the dataset table configuration.

    Returns:
        the parsed configuration or None on failure.
    """
    file_config = parse_dataset_file_config(table_config[TABLE_FILE])
    if file_config is None:
        return None

    # TODO parse this
    return DatasetTableConfig(
        table_config[TABLE_PRIMARY_KEY],
        table_config.get(TABLE_FOREIGN_KEYS),
        table_config[TABLE_COLUMNS],
        table_config[TABLE_NUM_RECORDS],
        file_config
    )


def parse_dataset_index_config(
        index_config: Dict[str, Any]) -> Optional[DatasetIndexConfig]:
    """Parse a dataset matrix' user/item index configuration.

    Args:
        index_config: the dataset matrix index configuration.

    Returns:
        the parsed configuration or None on failure.
    """
    # TODO parse this
    return DatasetIndexConfig(
        index_config.get(TABLE_FILE),
        index_config[TABLE_KEY],
        index_config[TABLE_NUM_RECORDS]
    )


def parse_dataset_matrix_config(
        matrix_config: Dict[str, Any]) -> Optional[DatasetMatrixConfig]:
    """Parse a dataset matrix configuration.

    Args:
        matrix_config: the dataset matrix configuration.

    Returns:
        the parsed configuration or None on failure.
    """
    # TODO parse this
    return DatasetMatrixConfig(
        parse_dataset_table_config(matrix_config[KEY_MATRIX]),
        matrix_config[KEY_RATING_MIN],
        matrix_config[KEY_RATING_MAX],
        matrix_config[KEY_RATING_TYPE],
        parse_dataset_index_config(matrix_config[KEY_IDX_USER]),
        parse_dataset_index_config(matrix_config[KEY_IDX_ITEM])
    )


def parse_dataset_config(
        dataset_config: Dict[str, Any]) -> Optional[DatasetConfig]:
    """Parse a dataset configuration.

    Args:
        dataset_config: the dataset configuration.

    Returns:
        the parsed configuration or None on failure.
    """
    events = {}
    if dataset_config.get(KEY_EVENTS) is not None:
        for table_name, table_config in dataset_config[KEY_EVENTS].items():
            config = parse_dataset_table_config(table_config)
            if config is None:
                continue

            events[table_name] = config

    matrices = {}
    if dataset_config.get(KEY_MATRICES) is not None:
        for matrix_name, matrix_config in dataset_config[KEY_MATRICES].items():
            config = parse_dataset_matrix_config(matrix_config)
            if config is None:
                continue

            matrices[matrix_name] = config

    tables = {}
    if dataset_config.get(KEY_TABLES) is not None:
        for table_name, table_config in dataset_config[KEY_TABLES].items():
            config = parse_dataset_table_config(table_config)
            if config is None:
                continue

            tables[table_name] = config

    return DatasetConfig(
        dataset_config[KEY_DATASET],
        events,
        matrices,
        tables
    )
