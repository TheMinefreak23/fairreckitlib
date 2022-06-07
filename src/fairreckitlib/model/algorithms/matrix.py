"""This module contains the matrix classes that can be used for algorithm training.

Classes:

    MatrixDataFrame: (base) matrix implementation for a pandas dataframe matrix.
    MatrixCSR: matrix implementation that uses a sparse CSR matrix.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import numpy as np
import pandas as pd
from scipy import sparse


class Matrix:
    """Base class for all train set matrices using a pandas dataframe.

    The intended use of the matrix class is to add an extra layer of
    abstraction to load a matrix into a specific format depending on
    the algorithm implementation that is used. Effectively this will
    reduce the memory usage of algorithms by loading the matrix directly
    into the expected format.

    Public methods:

    get_matrix
    get_items
    get_users
    get_user_rated_items
    knows_item
    knows_item_list
    knows_user
    knows_user_list
    """

    def __init__(self, file_path: str):
        """Construct the Matrix.

        The matrix is expected to be stored in a tab separated file without header,
        with the 'user', 'item', 'rating' columns in this order.

        Args:
            file_path: the file path to where the matrix is stored.

        Raises:
            FileNotFoundError: when the matrix file is not found.
        """
        self.matrix = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['user', 'item', 'rating']
        )
        self.users = self.matrix['user'].unique()
        self.items = self.matrix['item'].unique()

    def get_matrix(self) -> pd.DataFrame:
        """Get the matrix.

        Returns:
            the matrix dataframe.
        """
        return self.matrix

    def get_items(self) -> np.ndarray:
        """Get the (unique) items of the matrix.

        Returns:
            a list of unique item IDs.
        """
        return self.items

    def get_users(self) -> np.ndarray:
        """Get the (unique) users of the matrix.

        Returns:
            a list of unique user IDs.
        """
        return self.users

    def get_user_rated_items(self, user: int) -> np.ndarray:
        """Get the rated items for the specified user.

        Args:
            user: the user to get the rated items of.

        Raises:
            KeyError: when the user is not part of the matrix.

        Returns:
            a list of item IDs that are rated by the user.
        """
        if user not in self.users:
            raise KeyError('User is not part of the matrix')

        return self._get_user_rated_items(user)

    def _get_user_rated_items(self, user: int) -> np.ndarray:
        """Get the rated items for the specified user.

        Args:
            user: the user to get the rated items of.

        Returns:
            a list of item IDs that are rated by the user.
        """
        is_user = self.matrix['user'] == user
        return np.array(self.matrix.loc[is_user]['item'].tolist())

    def knows_item(self, item: int) -> bool:
        """Get if the specified item is known in the matrix.

        Args:
            item: the item ID to evaluate.

        Returns:
            whether the item ID is known.
        """
        return item in self.items

    def knows_item_list(self, items: pd.Series) -> pd.Series:
        """Get if the specified items are known in the matrix.

        Args:
            items: the item IDs to evaluate.

        Returns:
            a boolean series of the input showing whether each item is a known item ID.
        """
        return items.isin(self.items)

    def knows_user(self, user: int) -> bool:
        """Get if the specified user is known in the matrix.

        Args:
            user: the user ID to evaluate.

        Returns:
            whether the user ID is known.
        """
        return user in self.users

    def knows_user_list(self, users: pd.Series) -> pd.Series:
        """Get if the specified users are known in the matrix.

        Args:
            users: the user IDs to evaluate.

        Returns:
            a boolean series of the input showing whether each user is a known user ID.
        """
        return users.isin(self.users)


class MatrixCSR(Matrix):
    """Matrix implementation with a sparse CSR matrix."""

    def __init__(self, file_path: str):
        """Construct the CSR Matrix.

        The csr matrix is expected to be stored in a tab separated file without header,
        with the 'user', 'item', 'rating' columns in this order.
        The matrix is loaded into a dataframe and converted to a CSR matrix.

        Args:
            file_path: the file path to where the matrix is stored.
        """
        Matrix.__init__(self, file_path)
        self.matrix = sparse.csr_matrix(
            (self.matrix['rating'], (self.matrix['user'], self.matrix['item']))
        )

    def get_matrix(self) -> sparse.csr_matrix:
        """Get the matrix.

        Returns:
            the csr matrix.
        """
        return self.matrix

    def _get_user_rated_items(self, user: int) -> np.ndarray:
        """Get the rated items for the specified user.

        Args:
            user: the user to get the rated items of.

        Returns:
            a list of item IDs that are rated by the user.
        """
        return self.matrix[user].tocoo().col
