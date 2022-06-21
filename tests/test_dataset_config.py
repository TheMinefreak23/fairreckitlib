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
    TABLE_COMPRESSION, TABLE_ENCODING, TABLE_HEADER, TABLE_NUM_RECORDS, TABLE_SEP

STRING_LIST = ['a', 'b', 'c', 'd', 'e']

parser = DatasetConfigParser(True)


def test_parse_int(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a positive integer value from a dictionary."""
    int_key = 'int_key'

    assert not bool(parse_int({}, int_key, parse_event_dispatcher)), \
        'did not expect integer to be parsed that is not present'

    # test failure for incorrect cases, including 0 as the parser expect the to be integer > 0
    for int_value in [None, True, False, 'a', 0, 0.0, [], {}, {'set'}]:
        assert not bool(parse_int({int_key: int_value}, int_key, parse_event_dispatcher)), \
            'did not expect integer to be parsed from an incorrect input'

    # test success for correct input
    for i in range(1, 11):
        int_value = parse_int({int_key: i}, int_key, parse_event_dispatcher)
        assert not isinstance(int_value, bool), \
            'did not expect a boolean to be parsed'
        assert isinstance(int_value, int), \
            'expected an integer to be parsed'
        assert int_value == i, \
            'expected the parsed integer to be the same as the input value'


def test_parse_float(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a floating-point from a dictionary."""
    float_key = 'float_key'

    assert not bool(parse_float({}, float_key, parse_event_dispatcher)), \
        'did not expect float to be parsed that is not present'

    # test failure for incorrect cases
    for float_value in [None, True, False, 'a', 0, [], {}, {'set'}]:
        assert not bool(parse_float({float_key: float_value}, float_key, parse_event_dispatcher)), \
            'did not expect float to be parsed from an incorrect input'

    # test success for correct input
    for i in range(10):
        float_value = parse_float({float_key: float(i)}, float_key, parse_event_dispatcher)
        assert isinstance(float_value, float), \
            'expected a float to be parsed'
        assert float_value == float(i), \
            'expected the parsed integer to be the same as the input value'


def test_parse_string(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a string from a dictionary."""
    string_key = 'string_key'

    assert not bool(parse_string({}, string_key, parse_event_dispatcher)), \
        'did not expect string to be parsed that is not present'

    # test failure for incorrect cases
    for string_value in [None, True, False, 0.0, 0, [], {}, {'set'}]:
        assert not bool(parse_string(
            {string_key: string_value}, string_key, parse_event_dispatcher)), \
            'did not expect string to be parsed from an incorrect input'

    for string in STRING_LIST:
        # test failure for a value that is not included in a list of viable options
        one_of_list = [s for s in STRING_LIST if s != string]
        assert not bool(parse_string(
            {string_key: string}, string_key, parse_event_dispatcher, one_of_list=one_of_list)), \
            'did not expect string to be parsed that is not in the list of options'

        # test success for correct input
        string_value = parse_string({string_key: string}, string_key, parse_event_dispatcher)
        assert isinstance(string_value, str), \
            'expected a string to be parsed'
        assert string_value == string, \
            'expected the parsed string to be the same as the input string'


def test_parse_string_list(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a list of strings from a dictionary."""
    string_list_key = 'string_list_key'

    assert not bool(parse_string_list({}, string_list_key, 1, parse_event_dispatcher)), \
        'did not expect string list to be parsed that is not present'

    for string_list_value in [None, True, False, 0.0, 0, {}, {'set'}]:
        # test failure for incorrect cases
        assert not bool(parse_string_list(
            {string_list_key: string_list_value},
            string_list_key, 1, parse_event_dispatcher)), \
            'did not expect string list to be parsed from an incorrect input'

        # test failure for an incorrect value in a list of strings
        assert not bool(parse_string_list(
            {string_list_key: STRING_LIST + [string_list_value]},
            string_list_key, len(STRING_LIST) + 1, parse_event_dispatcher)), \
            'did not expect string list to be parsed from a list with one incorrect value'

    # test failure for minimum number of parsed strings to succeed
    assert not bool(parse_string_list(
        {string_list_key: []},
        string_list_key, 1, parse_event_dispatcher
    )), 'did not expect string list with at least one required value to be parsed for an empty list'

    string_list_value = parse_string_list(
        {string_list_key: STRING_LIST},
        string_list_key, len(STRING_LIST), parse_event_dispatcher
    )
    assert isinstance(string_list_value, list), \
        'expected a list to be parsed'
    assert len(string_list_value) == len(STRING_LIST), \
        'expected all strings in the list to be parsed'
    for string in string_list_value:
        assert isinstance(string, str), \
            'expected a string value to be parsed'


def test_parse_optional_bool(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing an optional boolean from a dictionary."""
    bool_key = 'bool_key'

    # test success when the bool is missing
    success, bool_value = parse_optional_bool({}, bool_key, parse_event_dispatcher)
    assert success, 'expected parsing to succeed for a boolean that is missing'
    assert not bool_value, 'expected False to be returned when it is missing'

    # test success when the bool is not missing or False
    success, bool_value = parse_optional_bool(
            {bool_key: True}, bool_key, parse_event_dispatcher)
    assert success, 'expected parsing to succeed for a boolean that is not missing or False'
    assert bool_value, 'expected the parsed boolean to be the same as the input'

    # test failure for incorrect cases
    for bool_value in ['a', 0.0, 0, [], {}, {'set'}]:
        success, bool_value = parse_optional_bool(
            {bool_key: bool_value}, bool_key, parse_event_dispatcher)
        assert not success, 'did not expect boolean to be parsed for an incorrect input'
        assert not bool(bool_value), 'expected None to be returned when the input is incorrect'


def test_parse_optional_string(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing an optional string from a dictionary."""
    string_key = 'string_key'

    success, string_value = parse_optional_string(
        {}, string_key, STRING_LIST, parse_event_dispatcher)
    assert success, 'expected parsing to succeed for a string that is missing'
    assert not bool(string_value), 'expected None to be returned when it is missing'

    # test failure for incorrect cases
    for string_value in [True, False, 0.0, 0, [], {}, {'set'}]:
        success, string_value = parse_optional_string(
            {string_key: string_value}, string_key, STRING_LIST, parse_event_dispatcher)

        assert not success, 'did not expect parsing to succeed for incorrect input'
        assert not bool(string_value), 'expected None to be returned when the input is incorrect'

    for string_value in STRING_LIST:
        # test failure for a value that is not included in a list of viable options
        string_value = [s for s in STRING_LIST if s != string_value]
        success, string_value = parse_optional_string(
            {string_key: string_value}, string_key, STRING_LIST, parse_event_dispatcher)
        assert not success, 'did not expect string to be parsed that is not in the list of options'
        assert not bool(string_value), 'expected None to be returned when the input is incorrect'


def test_parse_file_name(io_tmp_dir: str, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a file (name) from a dictionary."""
    file_key = 'file_key'

    for file_config in [{}, {file_key: None}]:
        # test failure for missing file name and missing file (on disk)
        success, file_name = parse_file_name(
            io_tmp_dir, file_config, file_key, parse_event_dispatcher)
        assert not success, 'did not expect file (name) to be parsed'
        assert not bool(file_name), 'expected None to be returned for a missing file (name)'

    for file_name in ['a', True, False, 0.0, 0, [], {}, {'set'}]:
        # test failure for various incorrect cases
        success, file_name = parse_file_name(
            io_tmp_dir, {file_key: file_name}, file_key, parse_event_dispatcher)
        assert not success, 'did not expect file (name) to be parsed for incorrect input'
        assert not bool(file_name), 'expected None to be returned for incorrect input'

    for file_name in STRING_LIST:
        # test success
        pd.DataFrame().to_csv(os.path.join(io_tmp_dir, file_name)) # store temp file on disk
        success, result_file_name = parse_file_name(
            io_tmp_dir, {file_key: file_name}, file_key, parse_event_dispatcher)
        assert success, 'expected file (name) to be parsed correctly'
        assert file_name == result_file_name, 'expected file name to be the same as the input'


def test_parse_rating_matrix(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing a rating matrix from a dictionary."""
    matrix_config = {}
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher)), \
        'did not expect ratings to be parsed for an empty configuration'

    matrix_config[KEY_RATING_MIN] = 0.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher)), \
        'did not expect ratings to be parsed for a minimum rating less or equal to zero'

    matrix_config[KEY_RATING_MIN] = 1.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher)), \
        'did not expect ratings to be parsed for a missing maximum rating'

    matrix_config[KEY_RATING_MAX] = 0.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher)), \
        'did not expect ratings to be parsed for a maximum rating less than minimum rating'

    matrix_config[KEY_RATING_MAX] = 1.0
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher)), \
        'did not expect ratings to be parsed for a missing rating type'

    matrix_config[KEY_RATING_TYPE] = 'unknown'
    assert not bool(parse_rating_matrix(matrix_config, parse_event_dispatcher)), \
        'did not expect ratings to be parsed for an unknown rating type'

    for rating_type in [DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT]:
        matrix_config[KEY_RATING_TYPE] = rating_type
        rating_matrix = parse_rating_matrix(matrix_config, parse_event_dispatcher)

        assert isinstance(rating_matrix, RatingMatrixConfig), \
            'expected RatingMatrixConfig to be parsed on correct input'
        assert matrix_config == rating_matrix.to_yml_format(), \
            'expected formatting RatingMatrixConfig to be the same as the original configuration'


def test_parse_file_options_config_minimal() -> None:
    """Test parsing minimal file options configuration from a dictionary."""
    file_options_config = {}

    # test failure for incorrect cases for all file options, which are all optional individually
    for invalid in ['a', True, False, 0.0, 0, [], {}, {'set'}]:
        # test separator option failure
        file_options_config[TABLE_SEP] = invalid
        assert not bool(parser.parse_file_options_config(file_options_config)), \
            'did not expect parsing to succeed for an incorrect separator value'

        # test compression option failure
        file_options_config[TABLE_SEP] = None
        file_options_config[TABLE_COMPRESSION] = invalid
        assert not bool(parser.parse_file_options_config(file_options_config)), \
            'did not expect parsing to succeed for an incorrect compression value'

        # test encoding option failure
        file_options_config[TABLE_COMPRESSION] = None
        file_options_config[TABLE_ENCODING] = invalid
        assert not bool(parser.parse_file_options_config(file_options_config)), \
            'did not expect parsing to succeed for an incorrect encoding value'

        file_options_config[TABLE_ENCODING] = None
        # skip boolean values which should succeed for optional booleans
        if not isinstance(invalid, bool):
            # test header option failure
            file_options_config[TABLE_HEADER] = invalid
            assert not bool(parser.parse_file_options_config(file_options_config)), \
                'did not expect parsing to succeed for an incorrect header value'

    # test success for an empty configuration
    file_options_config = {}
    parsed_options_config = parser.parse_file_options_config(file_options_config)
    assert isinstance(parsed_options_config, FileOptionsConfig), \
        'expected FileOptionsConfig to be parsed for an empty configuration'
    assert file_options_config == parsed_options_config.to_yml_format(), \
        'expected formatting FileOptionsConfig to be the same as the original configuration'


def test_parse_file_options_config_valid() -> None:
    """Test parsing valid file options configuration from a dictionary."""
    # test success for valid separator options
    for sep in VALID_SEPARATORS:
        file_options_config = {TABLE_SEP: sep}
        parsed_options_config = parser.parse_file_options_config(file_options_config)
        assert isinstance(parsed_options_config, FileOptionsConfig), \
            'expected FileOptionsConfig to be parsed for a configuration with a valid separator'
        assert parsed_options_config.sep == sep, \
            'expected parsed separator to be the same as the input'
        assert file_options_config == parsed_options_config.to_yml_format(), \
            'expected formatting FileOptionsConfig to be the same as the original configuration'

    # test success for valid compression options
    for compression in VALID_COMPRESSIONS:
        file_options_config = {TABLE_COMPRESSION: compression}
        parsed_options_config = parser.parse_file_options_config(file_options_config)
        assert isinstance(parsed_options_config, FileOptionsConfig), \
            'expected FileOptionsConfig to be parsed for a configuration with a valid compression'
        assert parsed_options_config.compression == compression, \
            'expected parsed compression to be the same as the input'
        assert file_options_config == parsed_options_config.to_yml_format(), \
            'expected formatting FileOptionsConfig to be the same as the original configuration'

    # test success for valid encoding options
    for encoding in VALID_ENCODINGS:
        file_options_config = {TABLE_ENCODING: encoding}
        parsed_options_config = parser.parse_file_options_config(file_options_config)
        assert isinstance(parsed_options_config, FileOptionsConfig), \
            'expected FileOptionsConfig to be parsed for a configuration with a valid encoding'
        assert parsed_options_config.encoding == encoding, \
            'expected parsed encoding to be the same as the input'
        assert file_options_config == parsed_options_config.to_yml_format(), \
            'expected formatting FileOptionsConfig to be the same as the original configuration'

    # test success for valid header option
    file_options_config = {TABLE_HEADER: True}
    parsed_options_config = parser.parse_file_options_config(file_options_config)
    assert isinstance(parsed_options_config, FileOptionsConfig), \
        'expected FileOptionsConfig to be parsed for a configuration with a valid header option'
    assert parsed_options_config.header, \
        'expected parsed header option to be the same as the input'
    assert file_options_config == parsed_options_config.to_yml_format(), \
            'expected formatting FileOptionsConfig to be the same as the original configuration'


def test_parse_dataset_file_config(io_tmp_dir: str) -> None:
    """Test parsing dataset file configuration from a dictionary."""
    assert not bool(parser.parse_dataset_file_config(io_tmp_dir, {})), \
        'did not expect parsing to succeed for an empty configuration'
    assert not bool(parser.parse_dataset_file_config(io_tmp_dir, {KEY_NAME: None})), \
        'did not expect parsing to succeed for an unknown file name'

    # store temporarily file on disk
    file_name = 'file_name'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, file_name))

    # test failure for when the file options are not correct
    assert not bool(parser.parse_dataset_file_config(
        io_tmp_dir, {KEY_NAME: file_name, TABLE_HEADER: ''})), \
        'did not expect parsing to succeed for an incorrect file options configuration'

    # test success
    file_config = {KEY_NAME: file_name}
    parsed_file_config = parser.parse_dataset_file_config(io_tmp_dir, file_config)
    assert isinstance(parsed_file_config, DatasetFileConfig), \
        'expected DatasetFileConfig to be parsed on correct input'
    assert parsed_file_config.name == file_name, \
        'expected parsed file name to be the same as the input'
    assert file_config == parsed_file_config.to_yml_format(), \
        'expected formatting DatasetFileConfig to be the same as the original configuration'


def test_parse_dataset_table_config(io_tmp_dir: str) -> None:
    """Test parsing dataset table configuration from a dictionary."""
    assert not bool(parser.parse_dataset_table_config(io_tmp_dir, {})), \
        'did not expect parsing to succeed for an empty configuration'

    # store temporarily file on disk
    file_name = 'file_name'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, file_name))

    # test failure for when the file configuration is not correct
    table_config = {TABLE_FILE: {KEY_NAME: file_name}}
    assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config)), \
        'did not expect parsing to succeed for an incorrect file configuration'

    # test failure for an empty list and integration failure after success
    for keys in [[], ['a']]:
        table_config[TABLE_PRIMARY_KEY] = keys
        assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config)), \
            'did not expect parsing to succeed for an empty list and integration'
        table_config[TABLE_FOREIGN_KEYS] = keys
        assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config)), \
            'did not expect parsing to succeed for an empty list and integration'
        table_config[TABLE_COLUMNS] = keys
        assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config)), \
            'did not expect parsing to succeed for an empty list and integration'

    table_config[TABLE_NUM_RECORDS] = 0
    assert not bool(parser.parse_dataset_table_config(io_tmp_dir, table_config)), \
        'did not expect parsing to succeed for a table with no records'

    table_config[TABLE_NUM_RECORDS] = 1
    parsed_config = parser.parse_dataset_table_config(io_tmp_dir, table_config)
    assert isinstance(parsed_config, DatasetTableConfig), \
        'expected DatasetTableConfig to be parsed on correct input'
    assert table_config == parsed_config.to_yml_format(), \
        'expected formatting DatasetTableConfig to be the same as the original configuration'


def test_parse_dataset_index_config(io_tmp_dir: str) -> None:
    """Test parsing dataset index configuration from a dictionary."""
    file_name = 'file_name'
    index_config = {TABLE_FILE: file_name}
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config)), \
        'did not expect parsing to succeed for an unknown file (name)'

    # store temporarily index file on disk
    save_array_to_hdf5(os.path.join(io_tmp_dir, file_name), [], file_name)
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config)), \
        'did not expect parsing to succeed for a missing index key'

    index_config[TABLE_KEY] = 'key'
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config)), \
        'did not expect parsing to succeed for indices with missing records'

    index_config[TABLE_NUM_RECORDS] = 0
    assert not bool(parser.parse_dataset_index_config(io_tmp_dir, index_config)), \
        'did not expect parsing to succeed for indices with no records'

    index_config[TABLE_NUM_RECORDS] = 1
    parsed_config = parser.parse_dataset_index_config(io_tmp_dir, index_config)
    assert isinstance(parsed_config, DatasetIndexConfig), \
        'expected DatasetIndexConfig to be parsed on correct input'
    assert index_config == parsed_config.to_yml_format()

    # index file is optional and should succeed when missing
    del index_config[TABLE_FILE]
    parsed_config = parser.parse_dataset_index_config(io_tmp_dir, index_config)
    assert isinstance(parsed_config, DatasetIndexConfig), \
        'expected DatasetIndexConfig to be parsed on correct input'
    assert index_config == parsed_config.to_yml_format(), \
        'expected formatting DatasetIndexConfig to be the same as the original configuration'


def test_parse_dataset_matrix_config(io_tmp_dir: str) -> None:
    """Test parsing dataset matrix configuration from a dictionary."""
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, {})), \
        'did not expect parsing to succeed for an empty configuration'

    # store temporarily file on disk
    matrix_file = 'matrix_file'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, matrix_file))
    # valid table configuration
    table_config = {
        TABLE_FILE: {KEY_NAME: matrix_file},
        TABLE_PRIMARY_KEY: ['primary', 'key'],
        TABLE_COLUMNS: ['column'],
        TABLE_NUM_RECORDS: 1
    }
    matrix_config = {KEY_MATRIX: table_config}
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config)), \
        'did not expect parsing to succeed with a missing user index configuration'

    matrix_config[KEY_IDX_USER] = {TABLE_KEY: 'user_id', TABLE_NUM_RECORDS: 1}
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config)), \
        'did not expect parsing to succeed with a missing item index configuration'

    matrix_config[KEY_IDX_ITEM] = {TABLE_KEY: 'item_id', TABLE_NUM_RECORDS: 1}
    assert not bool(parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config)), \
        'did not expect parsing to succeed with a missing rating matrix configuration'

    matrix_config[KEY_RATING_MIN] = 1.0
    matrix_config[KEY_RATING_MAX] = 1.0
    for rating_type in [DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT]:
        matrix_config[KEY_RATING_TYPE] = rating_type

        parsed_config = parser.parse_dataset_matrix_config(io_tmp_dir, matrix_config)
        assert isinstance(parsed_config, DatasetMatrixConfig), \
            'expected DatasetMatrixConfig to be parsed on correct input'
        assert matrix_config == parsed_config.to_yml_format(), \
            'expected formatting DatasetMatrixConfig to be the same as the original configuration'


def test_parse_dataset_config(io_tmp_dir: str) -> None:
    """Test parsing dataset configuration from a dictionary and yml file."""
    dataset_config = {}
    assert not bool(parser.parse_dataset_config(io_tmp_dir, dataset_config, [])), \
        'did not expect parsing to succeed for an empty configuration'

    dataset_name = 'dataset_name'
    dataset_config[KEY_DATASET] = dataset_name
    assert not bool(parser.parse_dataset_config(io_tmp_dir, dataset_config, [dataset_name])), \
        'did not expect parsing to succeed for a dataset that already exists'

    # test failure of (event) tables and matrices, but succeed the parsing in total
    for table_config in [None, True, False, 'a', 0.0, 0, [],  {}, {'set'}]:
        dataset_config[KEY_EVENTS] = table_config
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig), \
            'expected DatasetConfig to be parsed on correct input'
        assert len(parsed_config.events) == 0, \
            'did not expect any parsed event tables'

        dataset_config[KEY_MATRICES] = table_config
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig), \
            'expected DatasetConfig to be parsed on correct input'
        assert len(parsed_config.matrices) == 0, \
            'did not expect any parsed matrices'

        dataset_config[KEY_TABLES] = table_config
        parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
        assert isinstance(parsed_config, DatasetConfig), \
            'expected DatasetConfig to be parsed on correct input'
        assert len(parsed_config.tables) == 0, \
            'did not expect any parsed tables'

    # store temporarily file on disk
    table_name = 'table_name'
    table_file = 'table_file'
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, table_file))
    # valid table configuration
    table_config = {
        TABLE_FILE: {KEY_NAME: table_file},
        TABLE_PRIMARY_KEY: ['primary', 'key'],
        TABLE_COLUMNS: ['column'],
        TABLE_NUM_RECORDS: 1
    }

    # test success for an event table
    dataset_config[KEY_EVENTS] = {table_name: table_config}
    parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
    assert isinstance(parsed_config, DatasetConfig), \
        'expected DatasetConfig to be parsed on correct input'
    assert len(parsed_config.events) == 1, \
        'expected event table to be parsed'
    assert table_name in parsed_config.events, \
        'expected event table name to be present in the parsed configuration'
    assert dataset_config[KEY_EVENTS][table_name] == \
           parsed_config.events[table_name].to_yml_format(), \
        'expected formatting DatasetConfig to be the same as the original configuration'

    # test success for a matrix
    dataset_config[KEY_MATRICES] = {KEY_MATRIX}
    # valid matrix configuration, except for rating type
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
        assert isinstance(parsed_config, DatasetConfig), \
            'expected DatasetConfig to be parsed on correct input'
        assert len(parsed_config.matrices) == 1, \
            'expected matrix to be parsed'
        assert matrix_name in parsed_config.matrices, \
            'expected matrix name to be present in the parsed configuration'
        assert dataset_config[KEY_MATRICES][matrix_name] == \
               parsed_config.matrices[matrix_name].to_yml_format(), \
            'expected formatting DatasetConfig to be the same as the original configuration'

    # test success for a table
    dataset_config[KEY_TABLES] = {table_name: table_config}
    parsed_config = parser.parse_dataset_config(io_tmp_dir, dataset_config, [])
    assert isinstance(parsed_config, DatasetConfig), \
        'expected DatasetConfig to be parsed on correct input'
    assert len(parsed_config.tables) == 1, \
        'expected table to be parsed'
    assert table_name in parsed_config.tables, \
        'expected table name to be present in the parsed configuration'
    assert dataset_config[KEY_TABLES][table_name] == \
           parsed_config.tables[table_name].to_yml_format(), \
        'expected formatting DatasetConfig to be the same as the original configuration'

    # save configuration to disk
    yml_file_name = 'config'
    save_yml(os.path.join(io_tmp_dir, yml_file_name), dataset_config)

    # test failure for yml integration
    assert not bool(parser.parse_dataset_config_from_yml(
        io_tmp_dir, yml_file_name, [dataset_name])), \
        'expected parsing to fail from yml when the dataset already exists'

    # test success for yml integration
    parsed_config = parser.parse_dataset_config_from_yml(io_tmp_dir, yml_file_name, [])
    assert isinstance(parsed_config, DatasetConfig), \
        'expected DatasetConfig to be parsed on correct yml file input'
    assert dataset_config == parsed_config.to_yml_format(), \
        'expected formatting DatasetConfig to be the same as the original configuration'
