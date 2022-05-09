"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd

from ...utility import load_table
from ..dataset_config import DATASET_FILE
from ..dataset_config import DATASET_RATINGS_IMPLICIT
from ..dataset_config import DATASET_PREFIX
from ..dataset_table import create_table_config
from ..dataset_table import write_table
from .processor_base import DataProcessorBase


class DataProcessorLFM360K(DataProcessorBase):
    """DataProcessor for the LFM-360K dataset.

    Args:
        dataset_name(str): name of the dataset.
    """
    def __init__(self, dataset_name):
        DataProcessorBase.__init__(self, dataset_name, 'usersha1-artmbid-artname-plays.tsv')
        self.artist_mb_id = None

    def load_matrix(self, file_path):
        dataframe = load_table(
            file_path,
            ['user_sha', 'artist_mbID', 'artist_name', 'rating']
        )

        dataframe = dataframe[dataframe['user_sha'] != 'sep 20, 2008']

        dataframe['user_sha'] = dataframe['user_sha'].astype("category")
        dataframe['artist_name'] = dataframe['artist_name'].astype("category")
        dataframe['rating'] = dataframe['rating'].astype(float)

        dataframe = dataframe[dataframe['artist_name'].cat.codes >= 0]
        dataframe = dataframe[dataframe['rating'] > 0]

        self.user_list = list(dataframe['user_sha'].cat.categories)
        self.item_list = list(dataframe['artist_name'].cat.categories)

        dataframe['user'] = dataframe['user_sha'].cat.codes.copy()
        dataframe['item'] = dataframe['artist_name'].cat.codes.copy()

        self.data_matrix = dataframe[['user', 'item', 'rating']]

        self.artist_mb_id = dataframe[['artist_name', 'artist_mbID']]
        self.artist_mb_id = self.artist_mb_id.drop_duplicates().dropna()
        self.artist_mb_id = self.artist_mb_id.drop_duplicates(subset='artist_name')

        return True, DATASET_RATINGS_IMPLICIT

    def load_tables_config(self, dataset_dir, user_item_tables):
        return user_item_tables

    def load_user_table_config(self, dataset_dir):
        user_table_columns = [
            'user_sha',
            'user_gender',
            'user_age',
            'user_country',
            'user_signup'
        ]

        user_table = load_table(
            os.path.join(dataset_dir, 'usersha1-profile.tsv'),
            user_table_columns
        )

        user_sha_ids = pd.DataFrame(
            list(enumerate(self.user_list)),
            columns=['user_id', 'user_sha']
        )

        user_table = pd.merge(user_sha_ids, user_table, how='left', on='user_sha')
        self.user_list = None

        user_table['user_age'].mask(user_table['user_age'].lt(1), inplace=True)
        user_table['user_age'].mask(user_table['user_age'].gt(100), inplace=True)
        user_table['user_age'].fillna(-1.0, inplace=True)
        user_table['user_age'] = user_table['user_age'].astype(int)
        user_table['user_gender'].replace({'m': 'Male', 'f': 'Female'}, inplace=True)

        user_table_config = create_table_config(
            DATASET_PREFIX + self.dataset_name + '_users.tsv',
            ['user_id'],
            user_table_columns
        )

        write_table(
            user_table,
            dataset_dir,
            user_table_config[DATASET_FILE]
        )

        return 'user', user_table_config

    def load_item_table_config(self, dataset_dir):
        artist_table = pd.DataFrame(
            list(enumerate(self.item_list)),
            columns=['artist_id', 'artist_name']
        )
        self.item_list = None

        artist_table = pd.merge(artist_table, self.artist_mb_id, how='left', on='artist_name')
        artist_table['artist_mbID'].fillna(-1, inplace=True)

        artist_gender = pd.read_json(
            os.path.join(dataset_dir, 'lfm-360-gender.json'),
            orient='records',
            typ='series'
        ).to_frame()

        artist_gender.reset_index(inplace=True)
        artist_gender.rename(columns={'index': 'artist_mbID', 0: 'artist_gender'}, inplace=True)
        artist_gender.replace({'Not applicable': 'N/A'}, inplace=True)

        artist_table = pd.merge(artist_table, artist_gender, how='left', on='artist_mbID')

        artist_table_config = create_table_config(
            DATASET_PREFIX + self.dataset_name + '_artists.tsv',
            ['artist_id'],
            ['artist_name', 'artist_mbID', 'artist_gender']
        )

        write_table(
            artist_table,
            dataset_dir,
            artist_table_config[DATASET_FILE]
        )

        return 'artist', artist_table_config
