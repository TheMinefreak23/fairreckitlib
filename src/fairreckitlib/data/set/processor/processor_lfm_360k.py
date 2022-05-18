"""This modules contains the class to process the LastFM-360K dataset.

Classes:

    DataProcessorLFM360K: data processor implementation for the LFM-360K dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict, Tuple

import pandas as pd

from ...utility import load_table
from ..dataset import DATASET_RATINGS_IMPLICIT
from ..dataset_constants import DATASET_FILE, DATASET_PREFIX
from ..dataset_table import create_table_config, write_table
from .processor_base import DataProcessorBase


class DataProcessorLFM360K(DataProcessorBase):
    """DataProcessor for the LastFM-360K dataset."""

    def __init__(self, dataset_name: str):
        """Construct the LFM-360K DataProcessor.

        Args:
            dataset_name: name of the dataset.
        """
        DataProcessorBase.__init__(self, dataset_name, 'usersha1-artmbid-artname-plays.tsv')
        self.artist_mb_id = None

    def load_matrix(self, file_path: str) -> Tuple[bool, str]:
        """Load the matrix of the dataset.

        The user-item matrix is stored in a file that also contains a musicbrainzID.
        The users are hashes and the items are names, both are converted to integers
        to comply to the CSR compatible format. In addition, any rows that contain
        corrupt data are removed in the process.

        Args:
            file_path: the path to the matrix file.

        Returns:
            True, DATASET_RATINGS_IMPLICIT
        """
        dataframe = load_table(
            file_path,
            ['user_sha', 'artist_mbID', 'artist_name', 'rating']
        )

        # remove rows from a user that is not a hash
        dataframe = dataframe[dataframe['user_sha'] != 'sep 20, 2008']

        # map users/items to category and rating to be floating-point
        dataframe['user_sha'] = dataframe['user_sha'].astype("category")
        dataframe['artist_name'] = dataframe['artist_name'].astype("category")
        dataframe['rating'] = dataframe['rating'].astype(float)

        # remove rows that contain items that failed to map to category
        dataframe = dataframe[dataframe['artist_name'].cat.codes >= 0]
        # remove rows that have unusable ratings
        dataframe = dataframe[dataframe['rating'] > 0]

        # extract user/item indirection arrays
        self.user_list = list(dataframe['user_sha'].cat.categories)
        self.item_list = list(dataframe['artist_name'].cat.categories)

        # add the correct user/item integers
        dataframe['user'] = dataframe['user_sha'].cat.codes.copy()
        dataframe['item'] = dataframe['artist_name'].cat.codes.copy()

        # create matrix by removing other columns
        self.data_matrix = dataframe[['user', 'item', 'rating']]

        # extract artist name/musicbrainzID combinations
        self.artist_mb_id = dataframe[['artist_name', 'artist_mbID']]
        # remove duplicates combinations
        self.artist_mb_id = self.artist_mb_id.drop_duplicates().dropna()
        # remove duplicates where the artist has more than one musicbrainzID
        self.artist_mb_id = self.artist_mb_id.drop_duplicates(subset='artist_name')

        return True, DATASET_RATINGS_IMPLICIT

    def load_tables_config(self, dataset_dir: str, user_item_tables: Dict[str, Dict[str, Any]]):
        """Return the user and artist configurations."""
        return user_item_tables

    def load_user_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the user table of the dataset.

        The table is extended with unique user ids and changes the contents
        of the age and gender columns to be more user-friendly.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the user table.
        """
        user_table_columns = [
            'user_sha',
            'user_gender',
            'user_age',
            'user_country',
            'user_signup'
        ]

        # load original user table
        user_table = load_table(
            os.path.join(dataset_dir, 'usersha1-profile.tsv'),
            user_table_columns
        )

        # connect user id to sha and remove user indirection array
        user_sha_ids = pd.DataFrame(
            list(enumerate(self.user_list)),
            columns=['user_id', 'user_sha']
        )
        self.user_list = None

        # join user table with user ids
        user_table = pd.merge(user_sha_ids, user_table, how='left', on='user_sha')

        # mask user age not between 1-100, fill them with -1 and convert to int
        user_table['user_age'].mask(user_table['user_age'].lt(1), inplace=True)
        user_table['user_age'].mask(user_table['user_age'].gt(100), inplace=True)
        user_table['user_age'].fillna(-1.0, inplace=True)
        user_table['user_age'] = user_table['user_age'].astype(int)

        # convert gender to more user-friendly names
        user_table['user_gender'].replace({'m': 'Male', 'f': 'Female'}, inplace=True)

        # create user table configuration
        user_table_config = create_table_config(
            DATASET_PREFIX + self.dataset_name + '_users.tsv',
            ['user_id'],
            user_table_columns
        )

        # store the generated user table
        write_table(
            user_table,
            dataset_dir,
            user_table_config[DATASET_FILE]
        )

        return 'user', user_table_config

    def load_item_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the artist table of the dataset.

        Creates the artist table with the musicbrainzID and gender information when available.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the artist table.
        """
        artist_key = ['artist_id']
        artist_columns = ['artist_name']

        # connect artist id to name and remove item indirection array
        artist_table = pd.DataFrame(
            list(enumerate(self.item_list)),
            columns=artist_key + artist_columns
        )
        self.item_list = None

        # merge the artist musicbrainzID on name
        artist_table = pd.merge(artist_table, self.artist_mb_id, how='left', on='artist_name')
        artist_table['artist_mbID'].fillna(-1, inplace=True)
        artist_columns += ['artist_mbID']

        gender_json = 'lfm-360-gender.json'
        # TODO make lfm-360-gender.json optional
        if gender_json is not None:
            # read gender file
            artist_gender = pd.read_json(
                os.path.join(dataset_dir, gender_json),
                orient='records',
                typ='series'
            ).to_frame()

            # extract artist ids from index and rename
            artist_gender.reset_index(inplace=True)
            artist_gender.rename(columns={'index': 'artist_mbID', 0: 'artist_gender'}, inplace=True)
            artist_gender.replace({'Not applicable': 'N/A'}, inplace=True)

            # merge artists with gender and update columns
            artist_table = pd.merge(artist_table, artist_gender, how='left', on='artist_mbID')
            artist_columns += ['artist_gender']

        # create artist table configuration
        artist_table_config = create_table_config(
            DATASET_PREFIX + self.dataset_name + '_artists.tsv',
            artist_key,
            artist_columns
        )

        # store the generated artist table
        write_table(
            artist_table,
            dataset_dir,
            artist_table_config[DATASET_FILE]
        )

        return 'artist', artist_table_config
