"""This module tests the dataset wrapper functionality of the dataset configuration.

Functions:

    create_dataset_with_dummy_matrix: create dataset with a dummy matrix (configuration).
    test_dataset_constructor_error: test dataset construction error.
    test_dataset_available_columns: test the retrieval of available columns for a dataset matrix.
    test_dataset_available_event_tables: test the availability of the event tables of a dataset.
    test_dataset_available_matrices: test the availability of the matrices of a dataset.
    test_dataset_available_tables: test the availability of the tables of a dataset.
    test_dataset_get_matrices_info: test the retrieval of information from matrices of a dataset.
    test_dataset_get_matrix_config: test the retrieval of matrix configurations from a dataset.
    test_dataset_get_matrix_file_path: test the retrieval of matrix configurations from a dataset.
    test_dataset_get_table_config: test the retrieval of table configurations from a dataset.
    test_dataset_get_table_info: test the retrieval of information from tables of a dataset.
    test_dataset_load_matrix: test the matrix loading of a dataset.
    test_dataset_load_indices: test the user/item indices loading of a dataset.
    test_dataset_read_matrix: test reading the matrix tables of a dataset.
    test_dataset_read_table: test reading the available tables of a dataset.
    test_dataset_resolve_ids: test the index resolving functionality of a dataset.
    test_add_dataset_columns: test adding columns to a dataframe related to a dataset.
    assert_data_table_loading: assert table loading according to a table configuration.
    assert_data_table_and_columns: assert table (type), number of rows and requested columns.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Callable, List, Optional

import pytest

import numpy as np
import pandas as pd

from src.fairreckitlib.data.set.dataset import Dataset, add_dataset_columns
from src.fairreckitlib.data.set.dataset_config import \
    DatasetConfig, DatasetMatrixConfig, DatasetTableConfig, DatasetFileConfig
from src.fairreckitlib.data.set.dataset_constants import KEY_MATRIX, DATASET_SPLIT_DELIMITER
from src.fairreckitlib.data.set.dataset_registry import DataRegistry


def create_dataset_with_dummy_matrix(
        data_dir: str,
        dataset_name: str,
        matrix_name: str,
        matrix_file_config: DatasetFileConfig) -> Dataset:
    """Create dataset with a dummy matrix (configuration)."""
    return Dataset(
        data_dir,
        DatasetConfig(
            dataset_name,
            {}, # event tables not used
            {matrix_name: DatasetMatrixConfig(
                DatasetTableConfig(
                    [], # primary key not used
                    None, # foreign keys not used
                    [], # columns not used
                    0, # num_records not used
                    matrix_file_config
                ),
                None, # rating config not used
                None, # user config not used
                None  # item config not used
            )},
            {} # tables not used
        )
    )


def test_dataset_constructor_error(io_tmp_dir: str) -> None:
    """Test dataset construction error for an unknown data directory."""
    pytest.raises(
        IOError,
        Dataset,
        os.path.join(io_tmp_dir, 'unknown'),
        None # not used
    )


def test_dataset_available_columns(data_registry: DataRegistry) -> None:
    """Test the retrieval of available columns for a dataset matrix."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert len(dataset.get_available_columns('unknown')) == 0, \
            'did not expect available columns for an unknown dataset matrix'

        for matrix_name in dataset.get_available_matrices():
            available_columns = dataset.get_available_columns(matrix_name)

            assert isinstance(available_columns, dict), \
                'expected available columns to be a dictionary of available table columns'
            assert KEY_MATRIX in available_columns, \
                'expected matrix to be present in the available columns'

            matrix_config = dataset.get_matrix_config(matrix_name)

            for table_name in dataset.get_available_tables():
                table_config = dataset.get_table_config(table_name)

                related_keys = (
                    [matrix_config.user.key],
                    [matrix_config.item.key],
                    [matrix_config.user.key, matrix_config.item.key]
                )

                if table_config.primary_key in related_keys:
                    assert table_name in available_columns, \
                        'expected table related to the matrix keys to be present in the columns'
                    assert len(available_columns[table_name]) == len(table_config.columns), \
                        'expected all table columns to be available'
                    for column in table_config.columns:
                        assert column in available_columns[table_name], \
                            'expected table column to be available'


def test_dataset_available_event_tables(data_registry: DataRegistry) -> None:
    """Test the availability of the event tables of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        available_event_tables = dataset.get_available_event_tables()
        assert len(available_event_tables) == len(dataset.config.events), \
            'expected all dataset event tables in the configuration to be available'

        for table_name in available_event_tables:
            assert table_name in dataset.config.events, \
                'expected dataset event table to be present in the configuration'


def test_dataset_available_matrices(data_registry: DataRegistry) -> None:
    """Test the availability of the matrices of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        available_matrices = dataset.get_available_matrices()
        assert len(available_matrices) == len(dataset.config.matrices), \
            'expected all dataset matrices in the configuration to be available'

        for matrix_name in available_matrices:
            assert matrix_name in dataset.config.matrices, \
                'expected dataset matrix to be present in the configuration'


def test_dataset_available_tables(data_registry: DataRegistry) -> None:
    """Test the availability of the tables of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        available_tables = dataset.get_available_tables()
        assert len(available_tables) == len(dataset.config.tables), \
            'expected all dataset tables in the configuration to be available'

        for table_name in available_tables:
            assert table_name in dataset.config.tables, \
                'expected dataset table to be present in the configuration'


def test_dataset_get_matrices_info(data_registry: DataRegistry) -> None:
    """Test the retrieval of information from matrices of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        matrices_info = dataset.get_matrices_info()
        assert len(matrices_info) == len(dataset.get_available_matrices()), \
            'expected dataset matrix information for all available matrices'

        for matrix_name in dataset.get_available_matrices():
            assert matrix_name in matrices_info, \
                'expected dataset matrix information to be present'


def test_dataset_get_matrix_config(data_registry: DataRegistry) -> None:
    """Test the retrieval of matrix configurations from a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert not bool(dataset.get_matrix_config('unknown')), \
            'did not expect matrix configuration for an unknown matrix'

        for matrix_name in dataset.get_available_matrices():
            assert bool(dataset.get_matrix_config(matrix_name)), \
                'expected matrix configuration to be returned for a known matrix'


def test_dataset_get_matrix_file_path(data_registry: DataRegistry) -> None:
    """Test the retrieval of matrix configurations from a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert not bool(dataset.get_matrix_file_path('unknown')), \
            'did not expect matrix file path for an unknown matrix'

        for matrix_name in dataset.get_available_matrices():
            matrix_file_path = dataset.get_matrix_file_path(matrix_name)
            assert bool(matrix_file_path), \
                'expected matrix file path to be returned for a known matrix'
            assert os.path.isfile(matrix_file_path), \
                'expected matrix file path to be valid'


def test_dataset_get_table_config(data_registry: DataRegistry) -> None:
    """Test the retrieval of table configurations from a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert not bool(dataset.get_table_config('unknown')), \
            'did not expect table configuration for an unknown table'

        for table_name in dataset.get_available_tables():
            assert bool(dataset.get_table_config(table_name)), \
                'expected table configuration to be returned for a known table'


def test_dataset_get_table_info(data_registry: DataRegistry) -> None:
    """Test the retrieval of information from tables of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        table_info = dataset.get_table_info()
        assert len(table_info) == len(dataset.get_available_tables()), \
            'expected dataset table information for all available tables'

        for table_name in dataset.get_available_tables():
            assert table_name in table_info, \
                'expected dataset table information to be present'


def test_dataset_load_matrix(data_registry: DataRegistry) -> None:
    """Test the matrix loading of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert not bool(dataset.load_matrix('unknown')), \
            'did not expect unknown matrix to be loaded'

        for matrix_name in dataset.get_available_matrices():
            matrix_config = dataset.get_matrix_config(matrix_name)
            matrix = dataset.load_matrix(matrix_name)

            assert matrix is not None, 'expected a known matrix to be loaded'
            assert isinstance(matrix, pd.DataFrame), \
                'expected loaded matrix to be a dataframe'
            assert len(matrix) == matrix_config.table.num_records, \
                'expected loaded matrix to have all available rows'

            # verify standard format: 'user' 'item' 'rating' ['timestamp']
            assert 'user' in matrix.columns, \
                'expected \'user\' column to be present in the matrix'
            assert 'item' in matrix.columns, \
                'expected \'item\' column to be present in the matrix'
            assert 'rating' in matrix.columns, \
                'expected \'rating\' column to be present in the matrix'
            if len(matrix_config.table.columns) == 2:
                assert 'timestamp' in matrix.columns, \
                    'expected \'timestamp\' column to be present in the matrix'


def test_dataset_load_indices(data_registry: DataRegistry) -> None:
    """Test the user/item indices loading of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        # test failure for unknown matrix
        pytest.raises(KeyError, dataset.load_item_indices, 'unknown')
        pytest.raises(KeyError, dataset.load_user_indices, 'unknown')

        for matrix_name in dataset.get_available_matrices():
            matrix_config = dataset.get_matrix_config(matrix_name)

            item_indices = dataset.load_item_indices(matrix_name)
            if matrix_config.item.file_name is None:
                assert not bool(item_indices), \
                    'did not expect item indices to be loaded'
            else:
                assert isinstance(item_indices, np.ndarray), \
                    'expected item indices to be a numpy array'
                assert len(item_indices) == matrix_config.item.num_records, \
                    'expected item indices for all available items'

            user_indices = dataset.load_user_indices(matrix_name)
            if matrix_config.user.file_name is None:
                assert not bool(user_indices), \
                    'did not expect user indices to be loaded'
            else:
                assert isinstance(user_indices, np.ndarray), \
                    'expected user indices to be a numpy array'
                assert len(user_indices) == matrix_config.user.num_records, \
                    'expected user indices for all available users'


def test_dataset_read_matrix(data_registry: DataRegistry) -> None:
    """Test reading the matrix tables of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert not bool(dataset.read_matrix('unknown')), \
            'did not expect unknown matrix to be read'

        for matrix_name in dataset.get_available_matrices():
            assert_data_table_loading(
                dataset.read_matrix,
                matrix_name,
                dataset.get_matrix_config(matrix_name).table
            )


def test_dataset_read_table(data_registry: DataRegistry) -> None:
    """Test reading the available tables of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert not bool(dataset.read_matrix('unknown')), \
            'did not expect unknown matrix to be read'

        for table_name in dataset.get_available_tables():
            assert_data_table_loading(
                dataset.read_table,
                table_name,
                dataset.get_table_config(table_name)
            )


def test_dataset_resolve_ids(data_registry: DataRegistry) -> None:
    """Test the index resolving functionality of a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        for input_id in [0, [0]]:
            # test failure for resolving indices of an unknown matrix
            pytest.raises(KeyError, dataset.resolve_item_ids, 'unknown', input_id)
            pytest.raises(KeyError, dataset.resolve_user_ids, 'unknown', input_id)

        for matrix_name in dataset.get_available_matrices():
            matrix_config = dataset.get_matrix_config(matrix_name)

            resolve_tuples = [
                (matrix_config.item, dataset.resolve_item_ids),
                (matrix_config.user, dataset.resolve_user_ids)
            ]

            for index_config, resolve_ids in resolve_tuples:
                # test success for resolving the list of all ids
                num_ids = index_config.num_records
                resolved_ids = resolve_ids(matrix_name, list(range(num_ids)))
                assert isinstance(resolved_ids, (list, np.ndarray)), \
                    'expected a list of ids to be resolved'
                assert len(resolved_ids) == num_ids, \
                    'expected all ids to be resolved'

                # test success for resolving singular ids
                for idx in range(num_ids):
                    resolved_idx = resolve_ids(matrix_name, idx)
                    assert isinstance(resolved_idx, (int, np.int32, np.int64)), \
                        'expected integer idx to be resolved'


def test_add_dataset_columns(data_registry: DataRegistry) -> None:
    """Test adding columns to a dataframe related to a dataset."""
    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            available_columns = dataset.get_available_columns(matrix_name)

            all_columns = []
            for _, table_columns in available_columns.items():
                # gather accumulative of all available columns across tables
                all_columns += table_columns

                matrix = dataset.load_matrix(matrix_name)
                num_matrix_columns = len(matrix.columns)

                # test success for adding all columns of this table
                formatted_matrix = add_dataset_columns(dataset, matrix_name, matrix, table_columns)
                assert num_matrix_columns + len(table_columns) == len(formatted_matrix.columns), \
                    'expected all columns to be formatted to the matrix'
                for column in table_columns:
                    assert column in formatted_matrix, \
                        'expected column to be present in the formatted matrix'

                # test success for adding individual columns of this table
                for column in table_columns:
                    matrix = dataset.load_matrix(matrix_name)
                    num_matrix_columns = len(matrix.columns)

                    formatted_matrix = add_dataset_columns(dataset, matrix_name, matrix, [column])
                    assert num_matrix_columns + 1 == len(formatted_matrix.columns), \
                        'expected column to be formatted to the matrix'
                    assert column in formatted_matrix.columns, \
                        'expected column to be preset in the formatted matrix'

                # test success for adding all available columns for this matrix
                matrix = dataset.load_matrix(matrix_name)
                num_matrix_columns = len(matrix.columns)

                formatted_matrix = add_dataset_columns(dataset, matrix_name, matrix, all_columns)
                assert num_matrix_columns + len(all_columns) == len(formatted_matrix.columns), \
                    'expected all columns to be formatted to the matrix'
                for column in all_columns:
                    assert column in formatted_matrix, \
                        'expected column to be preset in the formatted matrix'


def assert_data_table_loading(
        load_table: Callable[[str, Optional[List[str]], Optional[int]], pd.DataFrame],
        table_name: str,
        table_config: DatasetTableConfig) -> None:
    """Assert loading the data table according to the accompanying table configuration."""
    table = load_table(table_name, None, None)
    assert_data_table_and_columns(table, table_config.num_records)

    # test reading all table columns (consisting of the primary key and available columns)
    available_columns = table_config.primary_key + table_config.columns
    table = load_table(table_name, available_columns, None)
    assert_data_table_and_columns(table, table_config.num_records, available_columns)
    assert len(table.groupby(table_config.primary_key).size()) == len(table), \
        'expected primary key to be unique in the table'

    # test reading individual keys that make up the primary key
    for key in table_config.primary_key:
        assert len(key.split(DATASET_SPLIT_DELIMITER)) == 2, \
            'expected to be able to split the key into two'

        available_columns = [key]
        table = load_table(table_name, available_columns, None)
        assert_data_table_and_columns(table, table_config.num_records, available_columns)
        assert not table[key].isna().any(), \
            'did not expect empty value in a table key column'

        # test reading individual table columns that are available
        for column in table_config.columns:
            assert len(column.split(DATASET_SPLIT_DELIMITER)) == 2, \
                'expected to be able to split the column into two'

            available_columns = [column]
            table = load_table(table_name, available_columns, None)
            assert_data_table_and_columns(table, table_config.num_records, available_columns)
            assert not table[column].isna().all(), \
                'expected table column to have at least one value'

            # test reading mix of key and table columns
            available_columns = [column, key] # order does not matter
            table = load_table(table_name, available_columns, None)
            assert_data_table_and_columns(table, table_config.num_records, available_columns)


def assert_data_table_and_columns(
        data_table: Any,
        num_records: int,
        requested_columns: List[str]=None) -> None:
    """Assert the data table (type), number of rows and requested columns if any."""
    assert data_table is not None, 'expected a data table to be read'
    assert isinstance(data_table, pd.DataFrame), \
        'expected read data table to be a dataframe'
    assert len(data_table) == num_records, \
        'expected read table to have all available rows'

    if requested_columns is None:
        return

    assert len(data_table.columns) == len(requested_columns), \
        'expected all requested columns to be available in the dataframe'
    for column in requested_columns:
        assert column in data_table, \
            'expected requested column to be present in the dataframe'
