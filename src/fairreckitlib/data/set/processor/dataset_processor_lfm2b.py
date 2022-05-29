"""This modules contains the class to process the LastFM-2B dataset.

Classes:

    DatasetProcessorLFM2B: data processor implementation for the LFM-2B dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..dataset_config import DatasetMatrixConfig, DatasetTableConfig, create_dataset_table_config
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_lfm import DatasetProcessorLFM


class DatasetProcessorLFM2B(DatasetProcessorLFM):
    """DatasetProcessor for the LastFM-2B dataset.

    The dataset can be downloaded from the website below.
    http://www.cp.jku.at/datasets/LFM-2b/

    Note that the compressed bz2 files can be used directly.
    The processor handles the following files:

    albums.tsv.bz2 (optional)
    artists.tsv.bz2 (optional)
    listening-counts.tsv.bz2 (required)
    listening-events.tsv.bz2 (required)
    spotify-uris.tsv.bz2 (optional)
    tracks.tsv.bz2 (optional)
    users.tsv.bz2 (optional)
    user_artist_playcount.tsv (required)
    """

    def create_listening_events_config(self) -> Optional[DatasetTableConfig]:
        """Create the listening event table configuration.

        Returns:
            the configuration of the listening event table.
        """
        return create_dataset_table_config(
            'listening-events.tsv.bz2',
            ['user_id', 'track_id', 'album_id'],
            ['timestamp'],
            compression='bz2',
            header=True
        )

    def create_user_table_config(self) -> DatasetTableConfig:
        """Create the user table configuration.

        Returns:
            the configuration of the user table.
        """
        return create_dataset_table_config(
            'users.tsv.bz2',
            ['user_id'],
            ['user_country', 'user_age', 'user_gender', 'user_creation time'],
            header=True,
            compression='bz2'
        )

    def get_matrix_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetMatrixConfig]]]]:
        """Get matrix configuration processors.

        Returns:
            a list containing the user-artist-count and user-track-count matrix processors.
        """
        return [
            ('user-artist-count', self.process_user_artist_matrix),
            ('user-track-count', self.process_user_track_matrix)
        ]

    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Returns:
            a list containing the album, artist, spotify, track and user table processors.
        """
        return DatasetProcessorLFM.get_table_configs(self) + [
            ('album', self.process_album_table),
            ('artist', self.process_artist_table),
            ('spotify', self.process_spotify_table),
            ('track', self.process_track_table)
        ]

    def process_album_table(self) -> Optional[DatasetTableConfig]:
        r"""Process the album table.

        The original file does not load correctly with pandas when splitting on
        newlines \n and \t tabs.

        Returns:
            the album table configuration or None on failure.
        """
        try:
            file_name, num_records = self.process_corrupt_table('albums')
        except FileNotFoundError:
            return None

        return create_dataset_table_config(
            file_name,
            ['album_id'],
            ['album_name', 'artist_name'],
            compression='bz2',
            num_records=num_records
        )

    def process_artist_table(self) -> Optional[DatasetTableConfig]:
        """Process the artist table.

        Returns:
            the artist table configuration or None on failure.
        """
        artist_table_config =  create_dataset_table_config(
            'artists.tsv.bz2',
            ['artist_id'],
            ['artist_name'],
            header=True,
            compression='bz2'
        )

        try:
            artist_table = artist_table_config.read_table(self.dataset_dir)
            artist_table_config.num_records = len(artist_table)
            return artist_table_config
        except FileNotFoundError:
            return None

    def process_corrupt_table(self, table_name: str) -> Tuple[str, int]:
        """Process a corrupt table that does not load correctly with pandas.

        Loading with the 'python-fwf' engine does not have issues, however the
        row values need to be manually split.
        """
        table_iterator = pd.read_table(
            os.path.join(self.dataset_dir, table_name + '.tsv.bz2'),
            header=0,
            encoding='utf-8',
            engine='python-fwf',
            names=['fwf'],
            iterator=True,
            chunksize=1000000
        )

        file_name = TABLE_FILE_PREFIX + self.dataset_name + '_' + table_name + '.tsv.bz2'
        file_path = os.path.join(self.dataset_dir, file_name)
        # remove existing file when present
        if os.path.isfile(file_path):
            os.remove(file_path)

        num_records = 0
        # process in chunks as splitting manually uses a lot of memory
        for _, dataframe in enumerate(table_iterator):
            dataframe = dataframe['fwf'].str.split('\t', expand=True)
            dataframe.to_csv(
                file_path,
                mode='a',
                sep='\t',
                index=False,
                header=False,
                compression='bz2'
            )
            num_records += len(dataframe)

        return file_name, num_records

    def process_spotify_table(self) -> Optional[DatasetTableConfig]:
        """Process the spotify table.

        Returns:
            the spotify table configuration or None on failure.
        """
        spotify_table_config =  create_dataset_table_config(
            'spotify-uris.tsv.bz2',
            ['track_id'],
            ['track_spotify-uri'],
            header=True,
            compression='bz2'
        )

        try:
            spotify_table = spotify_table_config.read_table(self.dataset_dir)
            spotify_table_config.num_records = len(spotify_table)
            return spotify_table_config
        except FileNotFoundError:
            return None

    def process_track_table(self) -> Optional[DatasetTableConfig]:
        r"""Process the track table.

        The original file does not load correctly with pandas when splitting on
        newlines \n and \t tabs.

        Returns:
            the track table configuration or None on failure.
        """
        try:
            file_name, num_records = self.process_corrupt_table('tracks')
        except FileNotFoundError:
            return None

        return create_dataset_table_config(
            file_name,
            ['track_id'],
            ['artist_name', 'track_name'],
            compression='bz2',
            num_records=num_records
        )

    def process_user_artist_matrix(self) -> Optional[DatasetMatrixConfig]:
        """Process the user-artist-count matrix.

        Returns:
            the matrix configuration or None on failure.
        """
        return self.process_matrix(create_dataset_table_config(
            'user_artist_playcount.tsv',
            ['user_id', 'artist_id'],
            ['matrix_count'],
            foreign_keys=['user_id', 'artist_id']
        ))

    def process_user_track_matrix(self) -> Optional[DatasetMatrixConfig]:
        """Process the user-track-count matrix.

        Returns:
            the matrix configuration or None on failure.
        """
        return self.process_matrix(create_dataset_table_config(
            'listening-counts.tsv.bz2',
            ['user_id', 'track_id'],
            ['matrix_count'],
            foreign_keys=['user_id', 'track_id'],
            compression='bz2',
            header=True
        ))
