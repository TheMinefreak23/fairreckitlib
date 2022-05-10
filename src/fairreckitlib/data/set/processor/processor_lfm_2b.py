"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...utility import load_table
from ..dataset import DATASET_RATINGS_IMPLICIT
from ..dataset_table import create_table_config
from .processor_base import DataProcessorBase


class DataProcessorLFM2B(DataProcessorBase):
    """DataProcessor for the LFM-2B dataset.

    Args:
        dataset_name(str): name of the dataset.
    """
    def __init__(self, dataset_name):
        DataProcessorBase.__init__(self, dataset_name, 'user_artist_playcount.tsv')

    def load_matrix(self, file_path):
        self.data_matrix = load_table(
            file_path,
            ['user', 'item', 'rating'],
        )

        return False, DATASET_RATINGS_IMPLICIT

    def load_tables_config(self, dataset_dir, user_item_tables):
        # TODO some of these tables have some issues with the header on the first line
        album_table_config = create_table_config(
            'albums.tsv.bz2',
            ['album_id'],
            ['album_name', 'artist_id']
        )

        count_table_config = create_table_config(
            'listening-counts.tsv.bz2',
            ['user_id', 'track_id'],
            ['play_count']
        )

        event_table_config = create_table_config(
            'listening-events.tsv.bz2',
            ['user_id', 'track_id'],
            ['album_id', 'timestamp']
        )

        spotify_table_config = create_table_config(
            'spotify-uris.tsv.bz2',
            ['track_id'],
            ['spotify_uri']
        )

        track_table_config = create_table_config(
            'tracks.tsv.bz2',
            ['track_id'],
            ['track_name', 'artist_id']
        )

        user_item_tables['album'] = album_table_config
        user_item_tables['count'] = count_table_config
        user_item_tables['event'] = event_table_config
        user_item_tables['spotify'] = spotify_table_config
        user_item_tables['track'] = track_table_config

        return user_item_tables

    def load_user_table_config(self, dataset_dir):
        user_table_config = create_table_config(
            'users.tsv.bz2',
            ['user_id'],
            ['user_country',
             'user_age',
             'user_gender',
             'user_creation time'
             ],
            header=True
        )

        return 'user', user_table_config

    def load_item_table_config(self, dataset_dir):
        artist_table_config =  create_table_config(
            'artists.tsv.bz2',
            ['artist_id'],
            ['artist_name'],
            header=True
        )

        return 'artist', artist_table_config
