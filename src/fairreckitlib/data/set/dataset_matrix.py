"""This module contains functionality to create matrices from listening events.

# TODO documentation

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

def create_track_matrix(dataframe) -> pd.DataFrame:
    """Create a user-track matrix from listening events.

    Use listening events to construct a user-track matrix with combined ratings.

    Args:
        dataset: the dataset with listening events and info about tracks.
    """
    # timestamps are not retained. possibly pass this as arg in the future.
    le_tracks = dataframe[['user','track','rating']]
    user_track = le_tracks.groupby(['user','track'], as_index=False).count()
    return user_track

def create_matrix(le_path: str, save_loc: str, item_type: str='track') -> pd.DataFrame:
    """Create a matrix from listening events.

    Using listening events, a user-item-rating matrix is created
    with all plays summed up to make a total rating per user-item pair.

    Args:
        le_path: the path to the listening event data, which should contain headers.
        item_type: the type of item you want in the matrix (e.g. track/artist/album)
        # TODO pass the dataset as arg for e.g. artist/album info
        # TODO use config

    Returns:
        matrix: the user-item matrix.
    """
    # reading chunks of the listening events
    num_chunks = 0
    for chunk in pd.read_csv(le_path, delimiter='\t', chunksize=10E6):
        # process and save the chunk
        # TODO change eval to something else, factory pattern?
        chunk.columns = ['user','rating','album','track','timestamp']
        chunk = eval('create_' + item_type + '_matrix(chunk)')
        chunk.to_csv(save_loc + '/chunk' + str(num_chunks) + '.tsv', index=False, sep='\t')
        num_chunks += 1
    matrix = pd.DataFrame({'user':pd.Series(dtype='int'),
                            item_type:pd.Series(dtype='int'),
                            'rating':pd.Series(dtype='float')})
    # when done with all the chunks, go through those chunks to sum those for the final result
    for num in range(num_chunks):
        # TODO maybe change location of where chunks are stored, for cleanup
        chunk = pd.read_csv(save_loc + '/chunk'+str(num)+'.tsv', delimiter='\t')
        matrix = matrix.append(chunk, ignore_index=True)
        matrix = matrix.groupby(['user','track','rating'], as_index=False).sum()

    unique_users = matrix['user'].unique()
    # users from 0...num_users
    matrix = pd.merge(
        matrix,
        pd.DataFrame(list(enumerate(unique_users)), columns=['user','user_id']),
        how='left',
        on='user'
    )
    # items from 0...num_items
    unique_items = matrix[item_type].unique()
    matrix = pd.merge(
        matrix,
        pd.DataFrame(list(enumerate(unique_items)), columns=['item',item_type]),
        how='left',
        on=item_type
    )
    matrix = matrix[['user',item_type,'rating']]
    matrix.to_csv(save_loc + '/user_' + item_type + '_matrix.zip',
                  index=False, header=False, sep='\t',
                  compression=dict(method='zip', archive_name='user_' + item_type + '_matrix.tsv'))
    return matrix
