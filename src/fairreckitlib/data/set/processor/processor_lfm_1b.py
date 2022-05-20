"""This modules contains the class to process the LastFM-1B dataset.

Classes:

    DataProcessorLFM1B: data processor implementation for the LFM-1B dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ...utility import convert_csr_to_coo, convert_coo_to_df
from ..dataset import DATASET_RATINGS_IMPLICIT
from ..dataset_constants import DATASET_FILE, DATASET_PREFIX
from ..dataset_table import create_table_config, read_table, write_table
from .processor_base import DataProcessorBase


class DataProcessorLFM1B(DataProcessorBase):
    """DataProcessor for the LastFM-1B dataset."""

    def __init__(self, dataset_name: str):
        """Construct the LFM-1B DataProcessor.

        Args:
            dataset_name: name of the dataset.
        """
        DataProcessorBase.__init__(self, dataset_name, 'LFM-1b_LEs.mat')

    def load_matrix(self, file_path: str) -> Tuple[bool, str]:
        """Load the matrix of the dataset.

        The user-item matrix is stored in a matlab file in CSR compatible format,
        together with the user and item indices. The matrix is converted
        to a dataframe and the indices for the indirection arrays are flattened.

        Args:
            file_path: the path to the matrix file.

        Returns:
            True, DATASET_RATINGS_IMPLICIT
        """
        # load matrix as described in the paper
        csr_matrix, idx_users, idx_artists = _load_lfm_1b_mat(file_path)

        # convert np.array of np.array to flat user/item lists
        self.user_list = list(map(lambda i: i[0], idx_users))
        self.item_list = list(map(lambda i: i[0], idx_artists))

        # create expected dataframe matrix
        self.data_matrix = convert_coo_to_df(
            convert_csr_to_coo(csr_matrix),
            'user', 'item', 'rating'
        )

        return True, DATASET_RATINGS_IMPLICIT

    def load_tables_config(self, dataset_dir: str, user_item_tables: Dict[str, Dict[str, Any]]):
        """Load the album, event and track tables of the dataset.

        Args:
            dataset_dir: directory where the dataset files are present.
            user_item_tables: dictionary containing the loaded user
                and artist table configurations.

        Returns:
            a dictionary with user, artist, album, event and track configurations.
        """
        album_table_config = create_table_config(
            'LFM-1b_albums.txt',
            ['album_id'],
            ['album_name', 'artist_id']
        )

        event_table_config = create_table_config(
            'LFM-1b_LEs.txt',
            ['user_id', 'artist_id'],
            ['album_id', 'track_id', 'timestamp']
        )

        track_table_config = create_table_config(
            'LFM-1b_tracks.txt',
            ['track_id'],
            ['track_name', 'artist_id']
        )

        user_item_tables['album'] = album_table_config
        user_item_tables['event'] = event_table_config
        user_item_tables['track'] = track_table_config

        return user_item_tables

    def load_user_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the user table of the dataset.

        Changes the contents of the gender column to be more user-friendly.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the user table.
        """
        user_table_config = create_table_config(
            'LFM-1b_users.txt',
            ['user_id'],
            ['user_country',
             'user_age',
             'user_gender',
             'user_plays',
             'user_registered'
             ],
            header=True
        )

        user_table = read_table(
            dataset_dir,
            user_table_config
        )

        # TODO what to do with LFM-1b_users_additional.txt

        # convert gender to more user-friendly names
        user_table['user_gender'].replace({'m': 'Male', 'f': 'Female', 'n': 'N/A'}, inplace=True)
        user_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_users.tsv'

        write_table(
            user_table,
            dataset_dir,
            user_table_config[DATASET_FILE]
        )

        return 'user', user_table_config

    def load_item_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the artist table of the dataset.

        Extends the artist table with gender information when available.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the artist table.
        """
        artist_table_config =  create_table_config(
            'LFM-1b_artists.txt',
            ['artist_id'],
            ['artist_name']
        )

        gender_json = 'lfm-gender.json'
        # TODO make lfm-gender.json optional
        if gender_json is not None:
            # read gender file
            artist_gender = pd.read_json(
                os.path.join(dataset_dir, gender_json),
                orient='records',
                typ='series'
            ).to_frame()

            # extract artist ids from index and rename
            artist_gender.reset_index(inplace=True)
            artist_gender.rename(columns={'index': 'artist_id', 0: 'artist_gender'}, inplace=True)
            artist_gender.replace({'Not applicable': 'N/A'}, inplace=True)

            # read original artist table
            artist_table = read_table(
                dataset_dir,
                artist_table_config
            )

            # merge artists with gender
            artist_table = pd.merge(artist_table, artist_gender, how='left', on='artist_id')

            # update artist table configuration
            artist_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_artists.tsv'
            artist_table_config['columns'] += ['artist_gender']

            # store the extended artist table
            write_table(
                artist_table,
                dataset_dir,
                artist_table_config[DATASET_FILE]
            )

        return 'artist', artist_table_config


def _load_lfm_1b_mat(file_path: str) -> Tuple[sparse.csr_matrix, np.array, np.array]:
    """Load the LFM-1B dataset from the matlab file.

    Args:
        file_path: the path to the matlab file.

    Returns:
        the matrix and user / artist indirection arrays.
    """
    with h5py.File(file_path, 'r') as mat_file:
        csr = sparse.csr_matrix((
            mat_file['/LEs/']["data"],
            mat_file['/LEs/']["ir"],
            mat_file['/LEs/']["jc"]
        )).transpose()
        idx_users = np.array(mat_file.get('idx_users')).astype(np.int64)
        idx_artists = np.array(mat_file.get('idx_artists')).astype(np.int64)
        return csr, idx_users, idx_artists
