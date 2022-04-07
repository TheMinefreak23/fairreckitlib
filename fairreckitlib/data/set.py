"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd
from scipy import sparse

DATASET_LFM_360K = 'LFM-360K'
DATASET_LFM_1B = 'LFM-1B'
DATASET_LFM_2B = 'LFM-2B'
DATASET_ML_100K = 'ML-100K'
DATASET_ML_25M = 'ML-25M'

DATASET_RATINGS_EXPLICIT = 'explicit'
DATASET_RATINGS_IMPLICIT = 'implicit'

class Dataset:

    def __init__(self, name, dir):
        self.name = name
        self.dir = dir
        # TODO get this data dynamically or load from config file
        if name == 'LFM-360K':
            self.num_users = 358868
            self.num_items = 292385
            self.num_pairs = 17535606
            self.ratings = DATASET_RATINGS_IMPLICIT
            self.min_rating = 0.0
            self.max_rating = 419157.0
            self.timestamp = False
        elif name == 'LFM-1B':
            self.num_users = 120175
            self.num_items = 585095
            self.num_pairs = 61534450
            self.ratings = DATASET_RATINGS_IMPLICIT
            self.min_rating = 1.0
            self.max_rating = 247612.0
            self.timestamp = False
        elif name == 'ML-100K':
            self.num_users = 943
            self.num_items = 1647
            self.num_pairs = 100000
            self.ratings = DATASET_RATINGS_EXPLICIT
            self.min_rating = 1.0
            self.max_rating = 5.0
            self.timestamp = False
        elif name == 'ML-25M':
            self.num_users = 162541
            self.num_items = 56623
            self.num_pairs = 25000095
            self.ratings = DATASET_RATINGS_EXPLICIT
            self.min_rating = 0.5
            self.max_rating = 5.0
            self.timestamp = True
        else:
            raise NotImplementedError()

    def get_info(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_pairs': self.num_pairs,
            'ratings': self.ratings,
            'min_rating': self.min_rating,
            'max_rating': self.max_rating,
            'timestamp': self.timestamp
        }

    def load_matrix_df(self):
        names = ['user', 'item', 'rating']
        if self.timestamp:
            names.append('timestamp')

        file_path = os.path.join(self.dir, self.name + '.tsv')
        return pd.read_csv(file_path, sep='\t', header=None, names=names)
