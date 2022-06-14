"""This module contains the parser for the dataset configuration and parser utility functions.

Classes:

    DatasetParser: dataset configuration parser.

Functions:

    parse_file_name: parse a file name from a configuration and verify existence on disk.
    parse_float: parse floating-point value from a configuration.
    parse_int: parse integer value from a configuration.
    parse_optional_bool: parse optional boolean value from a configuration.
    parse_optional_string: parse optional string value from a configuration.
    parse_rating_matrix: parse rating matrix configuration.
    parse_string: parse a string value from a configuration.
    parse_string_list: parse a list of strings from a configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import os.path
from typing import Any, Dict, List, Optional, Tuple

from ...core.core_constants import KEY_NAME
from ...core.events.event_dispatcher import EventDispatcher
from ...core.io.io_utility import load_yml
from ...core.parsing.parse_assert import \
    assert_is_type, assert_is_key_in_dict, assert_is_one_of_list
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs, print_parse_event
from .dataset_constants import KEY_DATASET, KEY_EVENTS, KEY_MATRICES, KEY_TABLES
from .dataset_constants import KEY_MATRIX, KEY_IDX_ITEM, KEY_IDX_USER
from .dataset_constants import KEY_RATING_MIN, KEY_RATING_MAX, KEY_RATING_TYPE
from .dataset_constants import TABLE_KEY, TABLE_PRIMARY_KEY, TABLE_FOREIGN_KEYS, TABLE_COLUMNS
from .dataset_constants import TABLE_FILE, TABLE_COMPRESSION, TABLE_ENCODING
from .dataset_constants import TABLE_HEADER, TABLE_INDEXED, TABLE_NUM_RECORDS, TABLE_SEP
from .dataset_config import DatasetIndexConfig, DatasetMatrixConfig, RatingMatrixConfig
from .dataset_config import DatasetConfig, DatasetFileConfig, DatasetTableConfig, FileOptionsConfig


class DatasetConfigParser:
    """Dataset Configuration Parser.

    Public methods:

    parse_dataset_config
    parse_dataset_config_from_yml
    """

    def __init__(self, verbose: bool):
        """Construct the DatasetConfigParser.

        Args:
            verbose: whether the parser should give verbose output.
        """
        self.verbose = verbose
        handle_parse_event = lambda parser, args: \
            print_parse_event(args) if parser.verbose else None

        self.event_dispatcher = EventDispatcher()
        self.event_dispatcher.add_listener(ON_PARSE, self, (handle_parse_event, None))

    def parse_dataset_config(
            self,
            data_dir: str,
            dataset_config: Dict[str, Any],
            available_datasets: List[str]) -> Optional[DatasetConfig]:
        """Parse a dataset configuration.

        Args:
            data_dir: the directory where the dataset is stored.
            dataset_config: the dataset configuration.
            available_datasets: a list of already available datasets.

        Returns:
            the parsed configuration or None on failure.
        """
        # attempt to parse the name of the dataset
        dataset_name = parse_string(
            dataset_config,
            KEY_DATASET,
            self.event_dispatcher
        )
        if dataset_name is None:
            return None

        # verify that the dataset name is not already present
        if dataset_name in available_datasets:
            self.event_dispatcher.dispatch(ParseEventArgs(
                ON_PARSE,
                'PARSE ERROR: dataset already exists: ' + dataset_name
            ))
            return None

        # attempt to parse the dataset (event) tables
        events = {}
        if dataset_config.get(KEY_EVENTS) is not None:
            for table_name, table_config in dataset_config[KEY_EVENTS].items():
                config = self.parse_dataset_table_config(data_dir, table_config)
                if config is None:
                    continue

                events[table_name] = config

        # attempt to parse the dataset (matrix) tables
        matrices = {}
        if dataset_config.get(KEY_MATRICES) is not None:
            for matrix_name, matrix_config in dataset_config[KEY_MATRICES].items():
                config = self.parse_dataset_matrix_config(
                    data_dir,
                    matrix_config
                )
                if config is None:
                    continue

                matrices[matrix_name] = config

        # attempt to parse the dataset (other) tables
        tables = {}
        if dataset_config.get(KEY_TABLES) is not None:
            for table_name, table_config in dataset_config[KEY_TABLES].items():
                config = self.parse_dataset_table_config(data_dir, table_config)
                if config is None:
                    continue

                tables[table_name] = config

        return DatasetConfig(
            dataset_name,
            events,
            matrices,
            tables
        )

    def parse_dataset_config_from_yml(
            self,
            data_dir: str,
            file_name: str,
            available_datasets: List[str]) -> Optional[DatasetConfig]:
        """Parse a dataset configuration.

        Args:
            data_dir: the directory where the dataset is stored.
            file_name: the name of the yml file without extension.
            available_datasets: a list of already available datasets.

        Returns:
            the parsed configuration or None on failure.
        """
        return self.parse_dataset_config(
            data_dir,
            load_yml(os.path.join(data_dir, file_name)),
            available_datasets
        )

    def parse_file_options_config(
            self,
            file_config: Dict[str, Any]) -> Optional[FileOptionsConfig]:
        """Parse a dataset file configuration.

        Args:
            file_config: the dataset file configuration.

        Returns:
            the parsed configuration or None on failure.
        """
        # attempt to parse the optional separator string
        success, file_sep = parse_optional_string(
            file_config,
            TABLE_SEP,
            [',', '|'],
            self.event_dispatcher
        )
        if not success:
            return None

        # attempt to parse the optional compression string
        success, file_compression = parse_optional_string(
            file_config,
            TABLE_COMPRESSION,
            ['bz2'],
            self.event_dispatcher
        )
        if not success:
            return None

        # attempt to parse the optional encoding string
        success, file_encoding = parse_optional_string(
            file_config,
            TABLE_ENCODING,
            ['utf-8', 'ISO-8859-1'],
            self.event_dispatcher
        )
        if not success:
            return None

        # attempt to parse the optional header boolean
        success, file_header = parse_optional_bool(
            file_config,
            TABLE_HEADER,
            self.event_dispatcher
        )
        if not success:
            return None

        # attempt to parse the optional indexed boolean
        success, file_indexed = parse_optional_bool(
            file_config,
            TABLE_INDEXED,
            self.event_dispatcher
        )
        if not success:
            return None

        return FileOptionsConfig(
            file_sep,
            file_compression,
            file_encoding,
            file_header,
            file_indexed
        )

    def parse_dataset_file_config(
            self,
            data_dir: str,
            file_config: Dict[str, Any]) -> Optional[DatasetFileConfig]:
        """Parse a dataset file configuration.

        Args:
            data_dir: the directory where the file is stored.
            file_config: the dataset file configuration.

        Returns:
            the parsed configuration or None on failure.
        """
        # attempt to parse the (required) file name
        success, file_name = parse_file_name(
            data_dir,
            file_config,
            KEY_NAME,
            self.event_dispatcher
        )
        if not success:
            return None

        # attempt to parse the file options
        file_options = self.parse_file_options_config(file_config)
        if file_options is None:
            return None

        return DatasetFileConfig(file_name, file_options)

    def parse_dataset_index_config(
            self,
            data_dir: str,
            index_config: Dict[str, Any]) -> Optional[DatasetIndexConfig]:
        """Parse a dataset matrix' user/item index configuration.

        Args:
            data_dir: the directory where the file is stored.
            index_config: the dataset matrix index configuration.

        Returns:
            the parsed configuration or None on failure.
        """
        # attempt to parse (optional) file name
        success, file_name = parse_file_name(
            data_dir,
            index_config,
            TABLE_FILE,
            self.event_dispatcher,
            required=False
        )
        if not success:
            return None

        # attempt to parse the key that is associated with the index
        file_key = parse_string(
            index_config,
            TABLE_KEY,
            self.event_dispatcher
        )
        if file_key is None:
            return None

        # attempt to parse the number of records in the file
        num_records = parse_int(
            index_config,
            TABLE_NUM_RECORDS,
            self.event_dispatcher
        )
        if num_records is None:
            return None

        return DatasetIndexConfig(file_name, file_key, num_records)

    def parse_dataset_matrix_config(
            self,
            data_dir: str,
            matrix_config: Dict[str, Any]) -> Optional[DatasetMatrixConfig]:
        """Parse a dataset matrix configuration.

        Args:
            data_dir: the directory where the dataset matrix is stored.
            matrix_config: the dataset matrix configuration.

        Returns:
            the parsed configuration or None on failure.
        """
        # attempt to parse the matrix table
        matrix_table = self.parse_dataset_table_config(data_dir,
                                                       matrix_config.get(KEY_MATRIX, {}))
        if matrix_table is None:
            return None

        # attempt to parse the matrix users
        matrix_users = self.parse_dataset_index_config(data_dir,
                                                       matrix_config.get(KEY_IDX_USER, {}))
        if matrix_users is None:
            return None

        # attempt to parse the matrix items
        matrix_items = self.parse_dataset_index_config(data_dir,
                                                       matrix_config.get(KEY_IDX_ITEM, {}))
        if matrix_items is None:
            return None

        # attempt to parse the matrix ratings
        matrix_ratings = parse_rating_matrix(
            matrix_config,
            self.event_dispatcher
        )
        if matrix_ratings is None:
            return None

        return DatasetMatrixConfig(
            matrix_table,
            matrix_ratings,
            matrix_users,
            matrix_items
        )

    def parse_dataset_table_config(
            self,
            data_dir: str,
            table_config: Dict[str, Any]) -> Optional[DatasetTableConfig]:
        """Parse a dataset table configuration.

        Args:
            data_dir: the directory where the table is stored.
            table_config: the dataset table configuration.

        Returns:
            the parsed configuration or None on failure.
        """
        file_config = self.parse_dataset_file_config(data_dir, table_config.get(TABLE_FILE, {}))
        if file_config is None:
            return None

        table_primary_key = parse_string_list(
            table_config,
            TABLE_PRIMARY_KEY,
            self.event_dispatcher
        )
        if table_primary_key is None:
            return None

        table_foreign_keys = None
        if TABLE_FOREIGN_KEYS in table_config:
            table_foreign_keys = parse_string_list(
                table_config,
                TABLE_FOREIGN_KEYS,
                self.event_dispatcher
            )
            if table_foreign_keys is None:
                return None

        table_columns = parse_string_list(
            table_config,
            TABLE_COLUMNS,
            self.event_dispatcher
        )
        if table_columns is None:
            return None

        table_num_records = parse_int(
            table_config,
            TABLE_NUM_RECORDS,
            self.event_dispatcher
        )
        if table_num_records is None:
            return None

        return DatasetTableConfig(
            table_primary_key,
            table_foreign_keys,
            table_columns,
            table_num_records,
            file_config
        )


def parse_file_name(
        data_dir: str,
        file_config: Dict[str, Any],
        file_key: str,
        event_dispatcher: EventDispatcher,
        *,
        required: bool=True) -> Tuple[bool, Optional[str]]:
    """Parse the file name from the configuration.

    In addition, when the file name is parsed correctly it is checked
    for existence in the specified data directory.

    Args:
        data_dir: the directory where the file is stored.
        file_config: the configuration dictionary to parse from.
        file_key: the key in the configuration that contains the file name.
        event_dispatcher: to dispatch the parse event on failure.
        required: whether the parsing is required to succeed.

    Returns:
        whether the parsing succeeded and the parsed file name or None on failure.
    """
    if required and not assert_is_key_in_dict(
        file_key,
        file_config,
        event_dispatcher,
        'PARSE ERROR: file configuration missing key \'' + file_key + '\''
    ): return False, None

    file_name = file_config.get(file_key)

    if file_name is not None:
        if not assert_is_type(
            file_name,
            str,
            event_dispatcher,
            'PARSE ERROR: file configuration contains invalid name'
        ): return False, None

        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            event_dispatcher.dispatch(ParseEventArgs(
                ON_PARSE,
                'PARSE ERROR: file configuration file name does not exist: ' + file_path
            ))
            return False, None

    return True, file_name


def parse_float(
        config: Dict[str, Any],
        float_key: str,
        event_dispatcher: EventDispatcher) -> Optional[float]:
    """Parse a float-point value from the configuration.

    Args:
        config: the configuration dictionary to parse from.
        float_key: the key in the configuration that contains the float-point value.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed float-point value or None on failure.
    """
    if not assert_is_key_in_dict(
        float_key,
        config,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + float_key + '\' value'
    ): return None

    float_value = config[float_key]

    if not assert_is_type(
        float_value,
        float,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + float_key + '\''
    ): return None

    return float_value


def parse_int(
        config: Dict[str, Any],
        int_key: str,
        event_dispatcher: EventDispatcher) -> Optional[int]:
    """Parse an integer value from the configuration.

    Args:
        config: the configuration dictionary to parse from.
        int_key: the key in the configuration that contains the integer value.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed integer value or None on failure.
    """
    if not assert_is_key_in_dict(
        int_key,
        config,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + int_key + '\' value'
    ): return None

    int_value = config[int_key]

    if not assert_is_type(
        int_value,
        int,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + int_key + '\''
    ): return None

    return int_value


def parse_optional_bool(
        config: Dict[str, Any],
        bool_key: str,
        event_dispatcher: EventDispatcher) -> Tuple[bool, Optional[bool]]:
    """Parse an optional boolean from the configuration.

    Args:
        config: the configuration dictionary to parse from.
        bool_key: the key in the configuration that contains the boolean.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        whether the parsing succeeded and the optional boolean value.
    """
    bool_value = config.get(bool_key)
    if bool_value is not None:
        if not assert_is_type(
            bool_value,
            bool,
            event_dispatcher,
            'PARSE ERROR: configuration contains invalid ' + bool_key + ' value'
        ): return False, None
    else:
        bool_value = False

    return True, bool_value


def parse_optional_string(
        config: Dict[str, Any],
        string_key: str,
        string_options: List[str],
        event_dispatcher: EventDispatcher) -> Tuple[bool, Optional[str]]:
    """Parse an optional string from a list of valid values from the configuration.

    Args:
        config: the configuration dictionary to parse from.
        string_key: the key in the configuration that contains the string.
        string_options: the options that are available for the string that is being parsed.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        whether the parsing succeeded and the optional string value.
    """
    string_value = config.get(string_key)
    if string_value is not None:
        if not assert_is_type(
            string_value,
            str,
            event_dispatcher,
            'PARSE ERROR: configuration contains invalid \'' + string_key + '\' value'
        ): return False, None

        if not assert_is_one_of_list(
            string_value,
            string_options,
            event_dispatcher,
            'PARSE ERROR: configuration contains invalid \'' + string_key + '\''
        ): return False, None

    return True, string_value


def parse_rating_matrix(
        matrix_config: Dict[str, Any],
        event_dispatcher: EventDispatcher) -> Optional[RatingMatrixConfig]:
    """Parse a rating matrix from the configuration.

    Args:
        matrix_config: the matrix configuration dictionary to parse from.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed string or None on failure.
    """
    rating_min = parse_float(
        matrix_config,
        KEY_RATING_MIN,
        event_dispatcher
    )
    if not rating_min:
        return None

    rating_max = parse_float(
        matrix_config,
        KEY_RATING_MAX,
        event_dispatcher
    )
    if not rating_max:
        return None

    rating_type = parse_string(
        matrix_config,
        KEY_RATING_TYPE,
        event_dispatcher
    )
    if not rating_type:
        return None

    return RatingMatrixConfig(rating_min, rating_max, rating_type)


def parse_string(
        config: Dict[str, Any],
        string_key: str,
        event_dispatcher: EventDispatcher) -> Optional[str]:
    """Parse a string from the configuration.

    Args:
        config: the configuration dictionary to parse from.
        string_key: the key in the configuration that contains the string.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed string or None on failure.
    """
    if not assert_is_key_in_dict(
        string_key,
        config,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + string_key + '\' value'
    ): return None

    string_value = config[string_key]

    if not assert_is_type(
        string_value,
        str,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + string_key + '\''
    ): return None

    return string_value


def parse_string_list(
        config: Dict[str, Any],
        string_list_key: str,
        event_dispatcher: EventDispatcher) -> Optional[List[str]]:
    """Parse a list of strings from the configuration.

    Args:
        config: the configuration dictionary to parse from.
        string_list_key: the key in the configuration that contains the string list.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed string list or None on failure.
    """
    if not assert_is_key_in_dict(
        string_list_key,
        config,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + string_list_key + '\' value'
    ): return None

    string_list = config[string_list_key]

    if not assert_is_type(
        string_list,
        list,
        event_dispatcher,
        'PARSE ERROR: configuration contains invalid \'' + string_list_key + '\''
    ): return None

    result_strings = []
    for string in string_list:
        if not assert_is_type(
            string,
            str,
            event_dispatcher,
            'PARSE ERROR: configuration list \'' + string_list_key + '\' contains invalid value'
        ): return None

        result_strings.append(string)

    return result_strings
