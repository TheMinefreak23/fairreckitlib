"""This module contains functionality to create a sample of an existing dataset.

Functions:

    create_dataset_sample: create a sample of a dataset.
    create_dataset_table_samples: create tables samples for a map of key indices.
    create_matrix_sample_config: create a sample matrix configuration.
    create_matrix_sample: create a sample of a dataset's matrix.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...core.io.io_utility import save_yml
from .dataset import Dataset
from .dataset_constants import DATASET_CONFIG_FILE
from .dataset_config import DatasetConfig, DatasetMatrixConfig, DatasetIndexConfig
from .dataset_config import DatasetTableConfig, create_dataset_table_config


def create_dataset_sample(
        output_dir: str,
        dataset: Dataset,
        num_users: int,
        num_items: int) -> Dataset:
    """Create a sample of the specified dataset.

    Look at the 'create_matrix_sample' function for specifics on how the
    matrices of the dataset are sampled. All tables, except the events, that are related
    to the user/item keys that are present in the sample matrices are sampled as well.
    The generated dataset sample is stored in the output directory before returning it.
    This function raises an IOError when the specified output directory already exists.

    Args:
        output_dir: the path to the directory where the dataset sample will be stored.
        dataset: the dataset to create a sample of.
        num_users: the number of users in the created sample matrices.
        num_items: the number of items in the created sample matrices.

    Returns:
        the resulting sample dataset.
    """
    sample_dir = os.path.join(output_dir, dataset.get_name() + '-Sample')
    if os.path.isdir(sample_dir):
        raise IOError('Failed to create sample, directory already exists.')

    os.mkdir(sample_dir)

    sample_matrices = {}
    key_id_map = {}

    for matrix_name in dataset.get_available_matrices():
        # create and add matrix sample
        sample_matrix_config = create_matrix_sample_config(
            sample_dir,
            dataset,
            matrix_name,
            num_users,
            num_items
        )
        sample_matrices[matrix_name] = sample_matrix_config

        # append user key indices to the key map
        user_keys = key_id_map.get(sample_matrix_config.user.key, [])
        user_indices = pd.Series(sample_matrix_config.user.load_indices(sample_dir))
        user_keys = pd.Series(user_keys).append(user_indices)
        key_id_map[sample_matrix_config.user.key] = user_keys.unique()

        # append item key indices to the key map
        item_keys = key_id_map.get(sample_matrix_config.item.key, [])
        item_indices = pd.Series(sample_matrix_config.item.load_indices(sample_dir))
        item_keys = pd.Series(item_keys).append(item_indices)
        key_id_map[sample_matrix_config.item.key] = item_keys.unique()

    # create sample tables for the key map that contains all the needed indices of all matrices
    sample_tables = create_dataset_table_samples(sample_dir, dataset, key_id_map)

    # create and save dataset configuration
    sample_dataset_config = DatasetConfig(dataset.get_name(), {}, sample_matrices, sample_tables)
    save_yml(os.path.join(sample_dir, DATASET_CONFIG_FILE), sample_dataset_config.to_yml_format())

    return Dataset(sample_dir, sample_dataset_config)


def create_dataset_table_samples(
        output_dir: str,
        dataset: Dataset,
        key_id_map: Dict[str, List[int]]) -> Dict[str, DatasetTableConfig]:
    """Create table samples for the specified dataset and key map.

    The key map is used to identify which tables of the dataset are sampled.
    A table is considered to be a candidate if the key in the map matches the
    primary key of the table. Any rows that do not contain the needed indices
    in the key map are filtered.

    Args:
        output_dir: the path to the directory where the sample tables will be stored.
        dataset: the dataset to create a sample tables from.
        key_id_map: a dictionary containing a table key paired with a list of indices
            that are related to these table keys.

    Returns:
        a dictionary with the resulting table sample configurations, keyed by table names.
    """
    sample_tables = {}

    for table_name in dataset.get_available_tables():
        table_config = dataset.get_table_config(table_name)

        table = dataset.read_table(table_name)
        table_modified = False

        for key_id, key_id_list in key_id_map.items():
            # filter unwanted table rows when the primary key matches
            if table_config.primary_key == [key_id]:
                table = table[table[key_id].isin(key_id_list)]
                table_modified = True

        if table_modified:
            # store the table sample
            sample_table_config = create_dataset_table_config(
                dataset.get_name() + '_' + table_name + '.tsv.bz2',
                table_config.primary_key,
                table_config.columns,
                compression='bz2',
                encoding=table_config.file.encoding,
                foreign_keys=table_config.foreign_keys,
                num_records=len(table)
            )
            sample_table_config.save_table(table, output_dir)
            # add sample table configuration
            sample_tables[table_name] = sample_table_config

    return sample_tables


def create_matrix_sample_config(
        output_dir: str,
        dataset: Dataset,
        matrix_name: str,
        num_users: int,
        num_items: int) -> Optional[DatasetMatrixConfig]:
    """Create a dataset matrix sample configuration.

    Look at the 'create_matrix_sample' function for specifics on how the
    matrix is sampled. The generated matrix and user/item indirection arrays are
    stored in the output directory and the corresponding configuration is returned.

    Args:
        output_dir: the path to the directory where the sample matrix will be stored.
        dataset: the dataset to create a sample matrix from.
        matrix_name: the name of the matrix to create a sample of.
        num_users: the number of users in the created sample matrix.
        num_items: the number of items in the created sample matrix.

    Returns:
        the sample matrix configuration or None when the specified matrix does not exist.
    """
    matrix_config = dataset.get_matrix_config(matrix_name)
    if matrix_config is None:
        return None

    sample, users, items = create_matrix_sample(dataset, matrix_name, num_users, num_items)

    # create the user indices config and save the array
    user_index_config = DatasetIndexConfig(
        matrix_name + '_user_indices.hdf5',
        matrix_config.user.key,
        len(users)
    )
    user_index_config.save_indices(output_dir, list(users))

    # create the item indices config and save the array
    item_index_config = DatasetIndexConfig(
        matrix_name + '_item_indices.hdf5',
        matrix_config.item.key,
        len(items)
    )
    item_index_config.save_indices(output_dir, list(items))

    # create the sample matrix table config and save the table
    sample_table_config = create_dataset_table_config(
        dataset.get_name() + '_' + matrix_name + '.tsv.bz2',
        matrix_config.table.primary_key,
        matrix_config.table.columns,
        compression='bz2',
        encoding=matrix_config.table.file.encoding,
        foreign_keys=matrix_config.table.foreign_keys,
        num_records=len(sample)
    )
    sample_table_config.save_table(sample, output_dir)

    return DatasetMatrixConfig(
        sample_table_config,
        sample[matrix_config.table.columns[0]].min(),
        sample[matrix_config.table.columns[0]].max(),
        matrix_config.rating_type,
        user_index_config,
        item_index_config
    )


def create_matrix_sample(
        dataset: Dataset,
        matrix_name: str,
        num_users: int,
        num_items: int) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """Create a sample for the specified matrix.

    Extracts a sample with the first occurring users and items until the
    specified amounts are reached, and therefore are only used as an indication.
    No additional users/items are generated when the dataset matrix has
    less available amounts than is specified. Moreover, due to the sparsity of the
    matrix it can turn out that the resulting matrix is very close, but not
    exactly the specified amounts.

    Args:
        dataset: the dataset to create a sample matrix from.
        matrix_name: the name of the matrix to create a sample of.
        num_users: the number of users in the created sample matrix.
        num_items: the number of items in the created sample matrix.

    Returns:
        the sample matrix, the unique user and unique item indices.
    """
    matrix_config = dataset.get_matrix_config(matrix_name)

    # clamp num users/items
    matrix_users = min(matrix_config.user.num_records, num_users)
    matrix_items = min(matrix_config.item.num_records, num_items)

    # prepare sample dataframe
    matrix_columns = matrix_config.table.primary_key + matrix_config.table.columns
    matrix_sample = pd.DataFrame(columns=matrix_columns)

    user_key = matrix_config.user.key
    item_key = matrix_config.item.key

    # create sample in chunks for very big matrices
    for _, matrix in enumerate(dataset.read_matrix(matrix_name, chunk_size=50000000)):
        matrix_sample = pd.concat([matrix_sample, matrix])
        if len(matrix_sample[user_key].unique()) > matrix_users and \
                len(matrix_sample[item_key].unique()) > matrix_items:
            break

    # users may not be number from 0...num_users
    unique_users = matrix_sample[user_key].unique()
    matrix_sample = pd.merge(
        matrix_sample,
        pd.DataFrame(list(enumerate(unique_users)), columns=['user',user_key]),
        how='left',
        on=user_key
    )
    # remove any users above the threshold
    matrix_sample = matrix_sample[matrix_sample['user'] < matrix_users]
    # recalculate the indirection array
    unique_users = dataset.resolve_user_ids(matrix_name, matrix_sample[user_key].unique().tolist())

    # items may not be number from 0...num_items
    unique_items = matrix_sample[item_key].unique()
    matrix_sample = pd.merge(
        matrix_sample,
        pd.DataFrame(list(enumerate(unique_items)), columns=['item',item_key]),
        how='left',
        on=item_key
    )
    # remove any items above the threshold
    matrix_sample = matrix_sample[matrix_sample['item'] < matrix_items]
    # recalculate and resolve the indirection array
    unique_items = dataset.resolve_item_ids(matrix_name, matrix_sample[item_key].unique().tolist())

    # create sample by removing the extra columns that were added
    matrix_sample = matrix_sample[['user', 'item'] + matrix_config.table.columns]

    return matrix_sample, unique_users, unique_items
