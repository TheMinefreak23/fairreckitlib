"""This modules contains the class to process the LastFM-1B dataset.

Classes:

    DatasetProcessorLFM1B: data processor implementation for the LFM-1B dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ..dataset_config import DATASET_RATINGS_IMPLICIT, RatingMatrixConfig
from ..dataset_config import \
    DatasetIndexConfig, DatasetMatrixConfig, DatasetTableConfig, create_dataset_table_config
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_lfm import DatasetProcessorLFM

ALL_MUSIC_GENRES = [
    'rnb', 'rap', 'electronic', 'rock', 'new age', 'classical', 'reggae', 'blues', 'country',
    'world', 'folk', 'easy listening', 'jazz', 'vocal', 'children\'s', 'punk', 'alternative',
    'spoken_word', 'pop', 'heavy metal'
]


class DatasetProcessorLFM1B(DatasetProcessorLFM):
    """DatasetProcessor for the LastFM-1B dataset.

    The dataset and UGP (user genre profile) can be downloaded from the website below.
    http://www.cp.jku.at/datasets/LFM-1b/

    The enriched artist gender information can be retrieved from:
    https://zenodo.org/record/3748787#.YowEBqhByUk

    The processor handles the following files:

    LFM-1b_albums.txt (optional)
    LFM-1b_artist_genres_allmusic.txt (optional)
    LFM-1b_artists.txt (optional)
    LFM-1b_LEs.mat (required)
    LFM-1b_LEs.txt (required)
    LFM-1b_tracks.txt (optional)
    LFM-1b_UGP_noPC_allmusic.txt (optional)
    LFM-1b_UGP_weightedPC_allmusic.txt (optional)
    LFM-1b_users.txt (optional)
    LFM-1b_users_additional.txt (optional)
    lfm-gender.json (optional)
    """

    def create_listening_events_config(self) -> Optional[DatasetTableConfig]:
        """Create the listening event table configuration.

        Returns:
            the configuration of the listening event table.
        """
        return create_dataset_table_config(
            'LFM-1b_LEs.txt',
            ['user_id', 'artist_id', 'album_id', 'track_id'],
            ['timestamp']
        )

    def create_user_table_config(self) -> DatasetTableConfig:
        """Create the user table configuration.

        Returns:
            the configuration of the user table.
        """
        return create_dataset_table_config(
            'LFM-1b_users.txt',
            ['user_id'],
            ['user_country', 'user_age', 'user_gender', 'user_plays', 'user_registered'],
            header=True
        )

    def get_matrix_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetMatrixConfig]]]]:
        """Get matrix configuration processors.

        Returns:
            a list containing the user-artist-count matrix processor.
        """
        return [('user-artist-count', self.process_user_artist_matrix)]

    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Returns:
            a list containing the album, allmusic genre, artist, track and user table processors.
        """
        return DatasetProcessorLFM.get_table_configs(self) + [
            ('album', self.process_album_table),
            ('allmusic genre', self.process_genres_allmusic),
            ('artist', self.process_artist_table),
            ('track', self.process_track_table),
            ('user additional', self.process_user_additional_table),
            ('user allmusic noPC', self.process_user_genre_allmusic_no_pc),
            ('user allmusic weightedPC', self.process_user_genre_allmusic_weighted_pc),
        ]

    def load_artist_gender_json(self) -> Optional[pd.DataFrame]:
        """Load the artist gender json file.

        Returns:
            the loaded artist id/gender table or None on failure.
        """
        try:
            gender_table = pd.read_json(
                os.path.join(self.dataset_dir, 'lfm-gender.json'),
                orient='index'
            )
            gender_table.reset_index(inplace=True)
            gender_table.rename(columns={'index': 'artist_id', 0: 'artist_gender'}, inplace=True)
            return gender_table
        except FileNotFoundError:
            return None

    def load_artist_genres_allmusic(self) -> Optional[pd.DataFrame]:
        """Load the artist allmusic genres file.

        Returns:
            the loaded artist name/genre table or None on failure.
        """
        try:
            genres = pd.read_csv(
                os.path.join(self.dataset_dir, 'LFM-1b_artist_genres_allmusic.txt'),
                sep='\t',
                names=['artist_name'] + [str(i) for i in range(0, len(ALL_MUSIC_GENRES))]
            )
        except FileNotFoundError:
            return None

        # remove duplicate rows where artist name is the same
        genres.drop_duplicates(subset='artist_name', inplace=True)
        # extract and drop artist name column
        artist_genres = pd.DataFrame(genres['artist_name'])
        genres.drop('artist_name', inplace=True, axis=1)

        # map allmusic genre id to genre name
        for col in genres:
            genres[col] = genres[col].map(lambda i: ALL_MUSIC_GENRES[int(i)], na_action='ignore')

        # add genres column
        artist_genres['artist_genres'] = genres.apply(lambda x: x.str.cat(sep='|'), axis=1)

        return artist_genres

    def process_album_table(self) -> Optional[DatasetTableConfig]:
        """Process the album table.

        Returns:
            the album table configuration or None on failure.
        """
        album_table_config = create_dataset_table_config(
            'LFM-1b_albums.txt',
            ['album_id'],
            ['album_name'],
            foreign_keys=['artist_id']
        )

        try:
            num_records = len(album_table_config.read_table(self.dataset_dir))
            album_table_config.num_records = num_records
            return album_table_config
        except FileNotFoundError:
            return None

    def process_artist_table(self) -> Optional[DatasetTableConfig]:
        """Process the artist table.

        Extends the table with artist gender and genres information when available.

        Returns:
            the artist table configuration or None on failure.
        """
        artist_table_config = create_dataset_table_config(
            'LFM-1b_artists.txt',
            ['artist_id'],
            ['artist_name']
        )

        try:
            artist_table = artist_table_config.read_table(self.dataset_dir)
        except FileNotFoundError:
            artist_table = pd.DataFrame()
            artist_table_config.columns.pop()

        # add artist gender when available
        gender_table = self.load_artist_gender_json()
        if gender_table is not None:
            # replace artist table when missing
            if len(artist_table) == 0:
                artist_table = gender_table
            # merge artist table with gender
            else:
                artist_table = pd.merge(artist_table, gender_table, how='left', on='artist_id')
            artist_table_config.columns += ['artist_gender']

        # no need to continue if the previous failed
        if len(artist_table) == 0:
            return None

        if 'artist_name' in artist_table_config.columns:
            # attempt to load artist name / genre table
            artist_genres = self.load_artist_genres_allmusic()
            if artist_genres is not None:
                # merge artist table with genres
                artist_table = pd.merge(artist_table, artist_genres, how='left', on='artist_name')
                artist_table_config.columns += ['artist_genres']

        artist_table_config.file.name = TABLE_FILE_PREFIX + self.dataset_name + '_artists.tsv.bz2'
        artist_table_config.file.compression = 'bz2'
        artist_table_config.num_records = len(artist_table)

        # store generated artist table
        artist_table_config.save_table(artist_table, self.dataset_dir)

        return artist_table_config

    def process_genres_allmusic(self) -> Optional[DatasetTableConfig]:
        """Process the allmusic genres table.

        Returns:
            the allmusic genres table configuration or None on failure.
        """
        genres_allmusic_table_config = create_dataset_table_config(
            'genres_allmusic.txt',
            ['allmusic_id'],
            ['allmusic_genre'],
            indexed=True
        )
        try:
            num_records = len(genres_allmusic_table_config.read_table(self.dataset_dir))
            genres_allmusic_table_config.num_records = num_records
            return genres_allmusic_table_config
        except FileNotFoundError:
            return None

    def process_track_table(self) -> Optional[DatasetTableConfig]:
        """Process the track table.

        Returns:
            the track table configuration or None on failure.
        """
        track_table_config = create_dataset_table_config(
            'LFM-1b_tracks.txt',
            ['track_id'],
            ['track_name'],
            foreign_keys=['artist_id']
        )

        try:
            num_records = len(track_table_config.read_table(self.dataset_dir))
            track_table_config.num_records = num_records
            return track_table_config
        except FileNotFoundError:
            return None

    def process_user_artist_matrix(self) -> Optional[DatasetMatrixConfig]:
        """Process the user-artist-count matrix.

        The user-item matrix is stored in a matlab file in CSR compatible format,
        together with the user and item indices. The matrix is converted
        to a dataframe and the indices for the indirection arrays are flattened.

        Returns:
            the matrix configuration or None on failure.
        """
        try:
            mat_file = os.path.join(self.dataset_dir, 'LFM-1b_LEs.mat')
            # load matrix as described in the paper
            csr_matrix, idx_users, idx_artists = _load_lfm_1b_mat(mat_file)
        except FileNotFoundError:
            return None

        matrix_name = 'user-artist-count'

        # create and save user indirection array
        user_list = list(map(lambda i: i[0], idx_users))
        user_index_config = DatasetIndexConfig(
            TABLE_FILE_PREFIX + self.dataset_name + '_' + matrix_name + '_user_indices.hdf5',
            'user_id',
            len(user_list)
        )
        user_index_config.save_indices(self.dataset_dir, user_list)

        # create and save artist indirection array
        artist_list = list(map(lambda i: i[0], idx_artists))
        artist_index_config = DatasetIndexConfig(
            TABLE_FILE_PREFIX + self.dataset_name + '_' + matrix_name + '_item_indices.hdf5',
            'artist_id',
            len(artist_list)
        )
        artist_index_config.save_indices(self.dataset_dir, artist_list)

        # convert csr to dataframe
        coo_matrix = pd.DataFrame.sparse.from_spmatrix(csr_matrix).sparse.to_coo()
        user_artist_matrix = pd.DataFrame()
        user_artist_matrix['user_id'] = coo_matrix.row
        user_artist_matrix['artist_id'] = coo_matrix.col
        user_artist_matrix['matrix_count'] = coo_matrix.data

        # create matrix table configuration
        user_artist_table_config = create_dataset_table_config(
            TABLE_FILE_PREFIX + self.dataset_name + '_' + matrix_name + '_matrix.tsv.bz2',
            ['user_id', 'artist_id'],
            ['matrix_count'],
            compression='bz2',
            foreign_keys=['user_id', 'artist_id'],
            num_records=len(user_artist_matrix)
        )

        # store the resulting matrix
        user_artist_table_config.save_table(user_artist_matrix, self.dataset_dir)

        return DatasetMatrixConfig(
            user_artist_table_config,
            RatingMatrixConfig(
                user_artist_matrix['matrix_count'].min(),
                user_artist_matrix['matrix_count'].max(),
                DATASET_RATINGS_IMPLICIT
            ),
            user_index_config,
            artist_index_config
        )

    def process_user_additional_table(self) -> Optional[DatasetTableConfig]:
        """Process the user additional table.

        Returns:
            the user additional table configuration or None on failure.
        """
        columns = [
            'user_novelty artist avg month',
            'user_novelty artist avg 6months',
            'user_novelty artist avg year',
            'user_mainstreaminess avg month',
            'user_mainstreaminess avg 6months',
            'user_mainstreaminess avg year',
            'user_mainstreaminess global',
            'user_count LEs',
            'user_count distinct tracks',
            'user_count distinct artists',
            'user_count LEs per week'
        ]

        for i in range(1, 8):
            columns += ['user_relative LE per weekday' + str(i)]
        for i in range(0, 24):
            columns += ['user_relative LE per hour' + str(i)]

        user_additional_table_config = create_dataset_table_config(
            'LFM-1b_users_additional.txt',
            ['user_id'],
            columns,
            header=True
        )

        try:
            num_records = len(user_additional_table_config.read_table(self.dataset_dir))
            user_additional_table_config.num_records = num_records
            return user_additional_table_config
        except FileNotFoundError:
            return None

    def process_user_genre_allmusic_no_pc(self) -> Optional[DatasetTableConfig]:
        """Process the user allmusic genre table.

        Returns:
            the user allmusic genre table configuration or None on failure.
        """
        columns = []
        for genre_name in ALL_MUSIC_GENRES:
            columns += ['noPC_' + genre_name]

        user_genre_allmusic_no_pc_config = create_dataset_table_config(
            'LFM-1b_UGP_noPC_allmusic.txt',
            ['user_id'],
            columns,
            header=True
        )
        try:
            num_records = len(user_genre_allmusic_no_pc_config.read_table(self.dataset_dir))
            user_genre_allmusic_no_pc_config.num_records = num_records
            return user_genre_allmusic_no_pc_config
        except FileNotFoundError:
            return None

    def process_user_genre_allmusic_weighted_pc(self) -> Optional[DatasetTableConfig]:
        """Process the user allmusic genre table with weighted play count.

        Returns:
            the user allmusic genre table configuration or None on failure.
        """
        columns = []
        for genre_name in ALL_MUSIC_GENRES:
            columns += ['weightedPC_' + genre_name]

        user_genre_allmusic_weighted_pc_config = create_dataset_table_config(
            'LFM-1b_UGP_noPC_allmusic.txt',
            ['user_id'],
            columns,
            header=True
        )
        try:
            num_records = len(user_genre_allmusic_weighted_pc_config.read_table(self.dataset_dir))
            user_genre_allmusic_weighted_pc_config.num_records = num_records
            return user_genre_allmusic_weighted_pc_config
        except FileNotFoundError:
            return None


def _load_lfm_1b_mat(file_path: str) -> Tuple[sparse.csr_matrix, np.array, np.array]:
    """Load the LFM-1B dataset from the matlab file.

    Args:
        file_path: the path to the matlab file.

    Returns:
        the matrix and user / artist indirection arrays.
    """
    with h5py.File(file_path, 'r') as mat_file:
        csr_matrix = sparse.csr_matrix((
            mat_file['/LEs/']["data"],
            mat_file['/LEs/']["ir"],
            mat_file['/LEs/']["jc"]
        )).transpose()
        idx_users = np.array(mat_file.get('idx_users')).astype(np.int64)
        idx_artists = np.array(mat_file.get('idx_artists')).astype(np.int64)
        return csr_matrix, idx_users, idx_artists
