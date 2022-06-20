"""This module tests the formatting and parsing of a dataset configuration.

Functions:

    test_parse_int: test parsing a positive integer value from a dictionary.
    test_parse_float: test parsing a floating-point from a dictionary.
    test_parse_string: test parsing a string from a dictionary.
    test_parse_string_list: test parsing a list of strings from a dictionary.
    test_parse_optional_bool: test parsing an optional boolean from a dictionary.
    test_parse_optional_string: test parsing an optional string from a dictionary.
    test_parse_file_name: test parsing a file (name) from a dictionary.
    test_parse_rating_matrix: test parsing a rating matrix from a dictionary.
    test_parse_file_options_config_minimal: test parsing minimal file options from a dictionary.
    test_parse_file_options_config_valid: test parsing valid file options from a dictionary.
    test_parse_dataset_file_config: test parsing dataset file from a configuration.
    test_parse_dataset_table_config: test parsing dataset table configuration from a dictionary.
    test_parse_dataset_index_config: test parsing dataset index configuration from a dictionary.
    test_parse_dataset_matrix_config: test parsing dataset matrix configuration from a dictionary.
    test_parse_dataset_config: test parsing dataset configuration from a dictionary and yml file.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd

from src.fairreckitlib.core.core_constants import KEY_NAME
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_utility import save_array_to_hdf5, save_yml
from src.fairreckitlib.data.set.dataset_config_parser import DatasetConfigParser, \
    parse_int, parse_float, parse_string, parse_string_list, \
    parse_optional_string, parse_optional_bool, parse_file_name, parse_rating_matrix, \
    VALID_SEPARATORS, VALID_COMPRESSIONS, VALID_ENCODINGS
from src.fairreckitlib.data.set.dataset_config import \
    DatasetIndexConfig, DatasetMatrixConfig, RatingMatrixConfig, \
    DatasetConfig, DatasetFileConfig, DatasetTableConfig, FileOptionsConfig, \
    DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT
from src.fairreckitlib.data.set.dataset_constants import \
    KEY_DATASET, KEY_EVENTS, KEY_MATRICES, KEY_TABLES, \
    KEY_MATRIX, KEY_IDX_ITEM, KEY_IDX_USER, KEY_RATING_MIN, KEY_RATING_MAX, KEY_RATING_TYPE, \
    TABLE_KEY, TABLE_PRIMARY_KEY, TABLE_FOREIGN_KEYS, TABLE_COLUMNS, TABLE_FILE, \
    TABLE_COMPRESSION, TABLE_ENCODING, TABLE_HEADER, TABLE_INDEXED, TABLE_NUM_RECORDS, TABLE_SEP

STRING_LIST = ['a', 'b', 'c', 'd', 'e']

parser = DatasetConfigParser(True)


def test_parse_int(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a positive integer value from a dictionary."""
    int_key = 'int_key'

    assert not bool(parse_int({}, int_key, parse_event_dispatcher))

    for int_value in [None, True, False, 'a', 0, 0.0, [], {}, {'set'}]:
        assert not bool(parse_int({int_key: int_value}, int_key, parse_event_dispatcher))

    for i in range(1, 11):
        int_value = parse_int({int_key: i}, int_key, parse_event_dispatcher)
        assert not isinstance(int_value, bool)
        assert isinstance(int_value, int)
        assert int_value == i


def test_parse_float(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a floating-point from a dictionary."""
    float_key = 'float_key'

    assert not bool(parse_float({}, float_key, parse_event_dispatcher))

    for float_value in [None, True, False, 'a', 0, [], {}, {'set'}]:
        assert not bool(parse_float({float_key: float_value}, float_key, parse_event_dispatcher))

    for i in range(10):
        float_value = parse_float({float_key: float(i)}, float_key, parse_event_dispatcher)
        assert isinstance(float_value, float)
        assert float_value == i


def test_parse_string(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a string from a dictionary."""
    string_key = 'string_key'

    assert not bool(parse_string({}, string_key, parse_event_dispatcher))

    for string_value in [None, True, False, 0.0, 0, [], {}, {'set'}]:
        assert not bool(parse_string(
            {string_key: string_value}, string_key, parse_event_dispatcher))

    for string in STRING_LIST:
        one_of_list = [s for s in STRING_LIST if s != string]
        assert not bool(parse_string(
            {string_key: string}, string_key, parse_event_dispatcher, one_of_list=one_of_list))

        string_value = parse_string({string_key: string}, string_key, parse_event_dispatcher)
        assert isinstance(string_value, str)
        assert string_value == string


def test_parse_string_list(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a list of strings from a dictionary."""
    string_list_key = 'string_list_key'

    assert not bool(parse_string_list({}, string_list_key, 1, parse_event_dispatcher))

    for string_list_value in [None, True, False, 0.0, 0, {}, {'set'}]:
        assert not bool(parse_string_list(
            {string_list_key: string_list_value},
            string_list_key, 1, parse_event_dispatcher))

        assert not bool(parse_string_list(
            {string_list_key: STRING_LIST + [string_list_value]},
            string_list_key, len(STRING_LIST) + 1, parse_event_dispatcher))

    assert not bool(parse_string_list(
        {string_list_key: []},
        string_list_key, 1, parse_event_dispatcher
    ))

    string_list_value = parse_string_list(
        {string_list_key: STRING_LIST},
        string_list_key, len(STRING_LIST), parse_event_dispatcher
    )
    assert isinstance(string_list_value, list)
    assert len(string_list_value) == len(STRING_LIST)
    for string in string_list_value:
        assert isinstance(string, str)


def test_parse_optional_bool(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing an optional boolean from a dictionary."""
    bool_key = 'bool_key'

    success, bool_value = parse_optional_bool({}, bool_key, parse_event_dispatcher)

    assert success
    assert not bool_value

    success, bool_value = parse_optional_bool(
            {bool_key: True}, bool_key, parse_event_dispatcher)

    assert success
    assert bool_value

    for bool_value in ['a', 0.0, 0, [], {}, {'set'}]:
        success, bool_value = parse_optional_bool(
            {bool_key: bool_value}, bool_key, parse_event_dispatcher)
        assert not success
        assert not bool(bool_value)


def test_parse_optional_string(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing an optional string from a dictionary."""
    string_key = 'string_key'

    success, string_value = parse_optional_string(
        {}, string_key, STRING_LIST, parse_event_dispatcher)

    assert success
    assert not bool(string_value)

    for string_value in [True, False, 0.0, 0, [], {}, {'set'}]:
        success, string_value = parse_optional_string(
            {string_key: string_value}, string_key, STRING_LIST, parse_event_dispatcher)

        assert not success
        assert not bool(string_value)

    for string_value in STRING_LIST:
        string_value = [s for s in STRING_LIST if s != string_value]
        success, string_value = parse_optional_string(
            {string_key: string_value}, string_key, STRING_LIST, parse_event_dispatcher)

        assert not success
        assert not bool(string_value)


def test_parse_file_name(io_tmp_dir: str, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a file (name) from a dictionary."""
    file_key = 'file_key'

    for file_config in [{}, {file_key: None}]:
        success, file_name = parse_file_name(
            io_tmp_dir, file_config, file_key, parse_event_dispatcher)

        assert not success
        assert not bool(file_name)

    for file_name in ['a', True, False, 0.0, 0, [], {}, {'set'}]:
        success, file_name = parse_file_name(
            io_tmp_dir, {file_key: file_name}, file_key, parse_event_dispatcher)

        assert not success
        assert not bool(file_name)

    for file_name in STRING_LIST:
        pd.DataFrame().to_csv(os.path.join(io_tmp_dir, file_name))
        success, result_file_name = parse_file_name(
            io_tmp_dir, {file_key: file_name}, file_key, parse_event_dispatcher)

        assert success
        assert file_name == result_file_name


def test_parse_rating_matrix(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a rating matrix from a dictionary."""
    matrix_config = {}
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher))

    matrix_config[KEY_RATING_MIN] = 0.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher))

    matrix_config[KEY_RATING_MIN] = 1.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher))

    matrix_config[KEY_RATING_MAX] = 0.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher))

    matrix_config[KEY_RATING_MAX] = 1.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher))

    matrix_config[KEY_RATING_TYPE] = 'unknown'
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher))

    for rating_type in [DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT]:
        matrix_config[KEY_RATING_TYPE] = rating_type
        rating_matrix = parse_rating_matrix(matrix_config, parse_event_dispatcher)

        assert isinstance(rating_matrix, RatingMatrixConfig)
        assert matrix_config == rating_matrix.to_yml_format()


def test_parse_file_options_config_minimal() -> None:
    """Test parsing minimal file options configuration from a dictionary."""
    file_options_config = {}

    for invalid in ['a', True, False, 0.0, 0, [], {}, {'set'}]:
        file_options_config[TABLE_SEP] = invalid
        assert not bool(parser.parse_file_options_config(file_options_config))

        file_options_config[TABLE_SEP] = None
        file_options_config[TABLE_COMPRESSION] = invalid
        assert not bool(parser.parse_file_options_config(file_options_config))

        file_options_config[TABLE_COMPRESSION] = None
        file_options_config[TABLE_ENCODING] = invalid
        assert not bool(parser.parse_file_options_config(file_options_config))

        file_options_config[TABLE_ENCODING] = None
        if not isinstance(invalid, bool):
            file_options_config[TABLE_HEADER] = invalid
            assert not bool(parser.parse_file_options_config(file_options_config))

            file_options_config[TABLE_INDEXED] = invalid
            assert not bool(parser.parse_file_options_config(file_options_config))

    file_options_config = {}
    parsed_options_config = parser.parse_file_options_config(file_options_config)
    assert isinstance(parsed_options_config, FileOptionsConfig)
    assert file_options_config == parsed_options_config.to_yml_format()


def test_parse_file_options_config_valid() -> None:
    """Test parsing valid file options configuration from a dictionary."""
    for sep in VALID_SEPARATORS:
        file_options_config = {TABLE_SEP: sep}
        parsed_options_config = parser.parse_file_options_config(file_options_config)
        assert isinstance(parsed_options_config, FileOptionsConfig)
        assert parsed_options_config.sep == sep
        assert file_options_config == parsed_options_config.to_yml_format()

    for compression in VALID_COMPRESSIONS:
        file_options_config = {TABLE_COMPRESSION: compression}
        parsed_options_config = parser.parse_file_options_config(file_options_config)
        assert isinstance(parsed_options_config, FileOptionsConfig)
        assert parsed_options_config.compression == compression
        assert file_options_config == parsed_options_config.to_yml_format()

    for encoding in VALID_ENCODINGS:
        file_options_config = {TABLE_ENCODING: encoding}
        parsed_options_config = parser.parse_file_options_config(file_options_config)
        assert isinstance(parsed_options_config, FileOptionsConfig)
        assert parsed_options_config.encoding == encoding
        assert file_options_config == parsed_options_config.to_yml_format()

    file_options_config = {TABLE_HEADER: True}
    parsed_options_config = parser.parse_file_options_config(file_options_config)
    assert isinstance(parsed_options_config, FileOptionsConfig)
    assert parsed_options_config.header
    assert file_options_config == parsed_options_config.to_yml_format()

    file_options_config = {TABLE_INDEXED: True}
    parsed_options_config = parser.parse_file_options_config(file_options_config)
    assert isinstance(parsed_options_config, FileOptionsConfig)
    assert parsed_options_config.indexed
    assert file_options_config == parsed_options_config.to_yml_format()


def test_parse_dataset_file_config(io_tmp_dir: str) -> None:
    """Test parsing dataset file configuration from a dictionary."""
    assert not bool(parser.parse_dataset_file_config(io_tmp_dir, {}))
    assert not bool(parser.parse_dataset_file_config(io_tmp_dir, {KEY_NAME: None}))

    file_name = 'file_name'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, file_name))

    for bool_key in [TABLE_HEADER, TABLE_INDEXED]:
        assert not bool(parser.parse_dataset_file_config(
            io_tmp_dir, {KEY_NAME: file_name, bool_key: ''}))

    file_config = {KEY_NAME: file_name}
    parsed_file_config = parser.parse_dataset_file_config(io_tmp_dir, file_config)
    assert isinstance(parsed_file_config, DatasetFileConfig)
    assert parsed_file_config.name == file_name
    assert file_config == parsed_file_config.to_yml_format()


def test_parse_dataset_table_config(io_tmp_dir: str) -> None:
    """Test parsing dataset table configuration from a dictionary."""
    assert not bool(parser.parse_dataset_table_config(io_tmp_dir, {}))

    file_name = 'file_name'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, file_name))

    table_config = {TABLE_FILE: {KEY_NAME: file_name}}
    assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config))

    for keys in [[], ['a']]:
        table_config[TABLE_PRIMARY_KEY] = keys
        assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config))

        table_config[TABLE_FOREIGN_KEYS] = keys
        assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config))

        table_config[TABLE_COLUMNS] = keys
        assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config))

    table_config[TABLE_NUM_RECORDS] = 0
    assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config))

    table_config[TABLE_NUM_RECORDS] = 1
    parsed_config = parser.parse_dataset_table_config(io_tmp_dir, table_config)
    assert isinstance(parsed_config, DatasetTableConfig)
    assert table_config == parsed_config.to_yml_format()


def test_parse_dataset_index_config(io_tmp_dir: str) -> None:
    """Test parsing dataset index configuration from a dictionary."""
    file_name = 'file_name'
    index_config = {TABLE_FILE: file_name}
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config))

    save_array_to_hdf5(os.path.join(io_tmp_dir, file_name), [], file_name)
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config))

    index_config[TABLE_KEY] = 'key'
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config))

    index_config[TABLE_NUM_RECORDS] = 0
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config))

    index_config[TABLE_NUM_RECORDS] = 1
    parsed_config = parser.parse_dataset_index_config(io_tmp_dir, index_config)
    assert isinstance(parsed_config, DatasetIndexConfig)
    assert index_config == parsed_config.to_yml_format()

    del index_config[TABLE_FILE]
    parsed_config = parser.parse_dataset_index_config(io_tmp_dir, index_config)
    assert isinstance(parsed_config, DatasetIndexConfig)
    assert index_config == parsed_config.to_yml_format()


def test_parse_dataset_matrix_config(io_tmp_dir: str) -> None:
    """Test parsing dataset matrix configuration from a dictionary."""
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, {}))

    matrix_file = 'matrix_file'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, matrix_file))
    table_config = {
        TABLE_FILE: {KEY_NAME: matrix_file},
        TABLE_PRIMARY_KEY: ['primary', 'key'],
        TABLE_COLUMNS: ['column'],
        TABLE_NUM_RECORDS: 1
    }
    matrix_config = {KEY_MATRIX: table_config}
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config))

    index_config = {TABLE_KEY: 'key', TABLE_NUM_RECORDS: 1}

    matrix_config[KEY_IDX_USER] = index_config
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config))

    matrix_config[KEY_IDX_ITEM] = index_config
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config))

    matrix_config[KEY_RATING_MIN] = 1.0
    matrix_config[KEY_RATING_MAX] = 1.0
    for rating_type in [DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT]:
        matrix_config[KEY_RATING_TYPE] = rating_type

        parsed_config = parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config)
        assert isinstance(parsed_config, DatasetMatrixConfig)
        assert matrix_config == parsed_config.to_yml_format()


def test_parse_dataset_config(io_tmp_dir: str) -> None:
    """Test parsing dataset configuration from a dictionary and yml file."""
    dataset_config = {}
    assert not bool(parser.parse_dataset_config(io_tmp_dir, dataset_config, []))

    dataset_name = 'dataset_name'
    dataset_config[KEY_DATASET] = dataset_name
    assert not bool(parser.parse_dataset_config(io_tmp_dir, dataset_config, [dataset_name]))

    for table_config in [None, True, False, 'a', 0.0, 0, [],  {}, {'set'}]:
        dataset_config[KEY_EVENTS] = table_config
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig)
        assert len(parsed_config.events) == 0

        dataset_config[KEY_MATRICES] = table_config
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig)
        assert len(parsed_config.events) == 0

        dataset_config[KEY_TABLES] = table_config
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig)
        assert len(parsed_config.events) == 0

    table_name = 'table_name'
    table_file = 'table_file'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, table_file))
    table_config = {
        TABLE_FILE: {KEY_NAME: table_file},
        TABLE_PRIMARY_KEY: ['primary', 'key'],
        TABLE_COLUMNS: ['column'],
        TABLE_NUM_RECORDS: 1
    }

    dataset_config[KEY_EVENTS] = {table_name: table_config}
    parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
    assert isinstance(parsed_config, DatasetConfig)
    assert len(parsed_config.events) == 1
    assert table_name in parsed_config.events
    assert dataset_config[KEY_EVENTS][table_name] == \
           parsed_config.events[table_name].to_yml_format()

    dataset_config[KEY_MATRICES] = {KEY_MATRIX}

    matrix_config = {
        KEY_MATRIX: table_config,
        KEY_IDX_USER: {TABLE_KEY: 'user_id', TABLE_NUM_RECORDS: 1},
        KEY_IDX_ITEM: {TABLE_KEY: 'item_id', TABLE_NUM_RECORDS: 1},
        KEY_RATING_MIN: 1.0,
        KEY_RATING_MAX: 1.0
    }
    for rating_type in [DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT]:
        matrix_config[KEY_RATING_TYPE] = rating_type
        matrix_name = 'matrix_name'
        dataset_config[KEY_MATRICES] = {matrix_name: matrix_config}
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig)
        assert len(parsed_config.matrices) == 1
        assert matrix_name in parsed_config.matrices
        assert dataset_config[KEY_MATRICES][matrix_name] == \
               parsed_config.matrices[matrix_name].to_yml_format()

    dataset_config[KEY_TABLES] = {table_name: table_config}
    parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
    assert isinstance(parsed_config, DatasetConfig)
    assert len(parsed_config.tables) == 1
    assert table_name in parsed_config.tables
    assert dataset_config[KEY_TABLES][table_name] == \
           parsed_config.tables[table_name].to_yml_format()

    yml_file_name = 'config'
    save_yml(os.path.join(io_tmp_dir, yml_file_name), dataset_config)
    assert not bool(parser.parse_dataset_config_from_yml(io_tmp_dir, yml_file_name, [dataset_name]))

    parsed_config = parser.parse_dataset_config_from_yml(io_tmp_dir, yml_file_name, [])
    assert isinstance(parsed_config, DatasetConfig)
    assert dataset_config == parsed_config.to_yml_format()
