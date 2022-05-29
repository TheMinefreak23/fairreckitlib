"""This module contains functionality to create matrices from listening events.

#TODO documentation

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

def create_matrix(le_path: str, item_type: str='track') -> pd.DataFrame:
    """Create a matrix from listening events.

    Args:
        le_path: the path to the listening event data.
        item_type: the item you want in the matrix (e.g. track/artist)
        # in future pass the dataset as arg for e.g. artist info

    Returns:
        matrix: the user-item matrix.
    """
    # reading in chunks of the listening events
    no = 0
    for chunk in pd.read_csv(le_path, delimiter='\t', chunksize=1E6):
        # process and save the chunk
        chunk = eval('create_' + item_type + '_matrix(chunk)')
        chunk.to_csv('chunks/chunk' + str(no) + '.tsv', index=False, sep='\t')
        no += 1
    # when done with all the chunks, go through those chunks to sum those for the final result
    matrix = pd.DataFrame({'user':[],item_type:[],'rating':[]})
    for n in range(no):
        # maybe change location of this, but not necessary to pass this as arg, it's just internal
        chunk = pd.read_csv('chunks/chunk'+str(n)+'.tsv', delimiter='\t')
        matrix = matrix.append(chunk, ignore_index=True)
        # this needs to be changed to a sum instead of count for final ratings
        matrix = eval('create_', item_type, '_matrix(matrix)')

    matrix.to_csv('chunks/1matrix.tsv', index=False, sep='\t')
    return matrix
