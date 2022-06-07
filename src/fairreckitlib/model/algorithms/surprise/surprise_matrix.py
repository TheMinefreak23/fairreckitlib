"""This module contains a matrix implementation for the surprise package.

Classes:

    MatrixSurprise: the surprise matrix class.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Tuple

import numpy as np
import surprise

from ..matrix import Matrix


class MatrixSurprise(Matrix):
    """Matrix implementation with a surprise.Trainset."""

    def __init__(self, file_path: str, rating_scale: Tuple[float, float]):
        """Construct the CSR Matrix.

        The surprise matrix is expected to be stored in a tab separated file without header,
        with the 'user', 'item', 'rating' columns in this order.
        The matrix is loaded into a surprise dataset and converted to a full train set.

        Args:
            file_path: the file path to where the matrix is stored.
            rating_scale: the minimum and maximum rating in the loaded set.
        """
        Matrix.__init__(self, file_path)
        reader = surprise.Reader(rating_scale=rating_scale)
        self.matrix = surprise.Dataset.load_from_df(self.matrix, reader)
        self.matrix = self.matrix.build_full_trainset()

    def get_matrix(self) -> surprise.Trainset:
        """Get the matrix.

        Returns:
            the surprise.Trainset matrix.
        """
        return self.matrix

    def _get_user_rated_items(self, user: int) -> np.ndarray:
        """Get the rated items for the specified user.

        The user ratings are stored in the 'ur' dictionary of the surprise.Trainset.

        Args:
            user: the user to get the rated items of.

        Returns:
            a list of item IDs that are rated by the user.
        """
        uid = self.matrix.to_inner_uid(user)
        return np.array([self.matrix.to_raw_iid(i) for (i, _) in self.matrix.ur[uid]])
