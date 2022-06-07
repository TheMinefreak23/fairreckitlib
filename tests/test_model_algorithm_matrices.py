"""This module tests the interface of the algorithm matrices.

Classes:

    DummyMatrix: matrix that is used to test algorithm training type errors.

Functions:

    assert_algo_matrix_items: assert the item functionality of a matrix.
    assert_algo_matrix_users: assert the user functionality of a matrix.
    assert_algo_matrix_user_rated_items: assert the user rated items functionality of a matrix.
    create_algo_matrix: create matrices for all algorithm APIs.
    test_algo_matrix: test if the matrix is sufficient for testing.
    test_algo_matrix_interface: test matrices for all algorithm APIs.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd
import pytest

from src.fairreckitlib.core.core_constants import IMPLICIT_API, LENSKIT_API, SURPRISE_API
from src.fairreckitlib.model.algorithms.matrix import Matrix, MatrixCSR
from src.fairreckitlib.model.algorithms.surprise.surprise_matrix import MatrixSurprise

MATRIX_DIR = os.path.join('tests', 'files')
MATRIX_FILE = os.path.join(MATRIX_DIR, 'algorithm_matrix.tsv')
MATRIX_RATING_SCALE = (1.0, 5.0)

# matrix dataframe and unique users/items for comparison in matrix/algorithm tests
algo_matrix_df = pd.read_table(MATRIX_FILE, header=None, names=['user', 'item', 'rating'])
algo_matrix_users = algo_matrix_df['user'].unique()
algo_matrix_items = algo_matrix_df['item'].unique()


class DummyMatrix(Matrix):
    """Dummy matrix used to test algorithm training errors."""

    def __init__(self):
        """Construct dummy matrix."""
        Matrix.__init__(self, MATRIX_FILE)

    def get_matrix(self) -> None:
        """Get the matrix returns None will fail for any algorithm."""
        return None


def assert_algo_matrix_items(matrix: Matrix) -> None:
    """Assert the item functionality of the specified matrix."""
    assert len(matrix.get_items()) == len(algo_matrix_items), \
        'expected matrix unique items to be the same as in the original matrix.'

    item_min = algo_matrix_items.min()
    item_max = algo_matrix_items.max()

    assert not matrix.knows_item(item_min - 1), \
        'did not expect matrix to know unknown item.'
    assert not matrix.knows_item(item_max + 1), \
        'did not expect matrix to know unknown item.'
    assert not matrix.knows_item_list(pd.Series([item_min - 1, item_max + 1])).any(), \
        'did not expect matrix to know unknown items.'

    assert matrix.knows_item_list(pd.Series(algo_matrix_items)).all(), \
        'expected original matrix items to be known in the matrix.'
    for item in algo_matrix_items:
        assert matrix.knows_item(item), \
            'expected original matrix item to be known in the matrix.'

    for item in matrix.get_items():
        assert item in algo_matrix_items, \
            'expected item to be present in the original matrix items.'


def assert_algo_matrix_users(matrix: Matrix) -> None:
    """Assert the user functionality of the specified matrix."""
    assert len(matrix.get_users()) == len(algo_matrix_users), \
        'expected matrix unique users to be the same as in the original matrix.'

    user_min = algo_matrix_users.min()
    user_max = algo_matrix_users.max()

    assert not matrix.knows_user(user_min - 1), \
        'did not expect matrix to know unknown user.'
    assert not matrix.knows_user(user_max + 1), \
        'did not expect matrix to know unknown user.'
    assert not matrix.knows_user_list(pd.Series([user_min - 1, user_max + 1])).any(), \
        'did not expect matrix to know unknown users.'

    assert matrix.knows_user_list(pd.Series(algo_matrix_users)).all(), \
        'expected original matrix users to be known in the matrix.'
    for user in algo_matrix_users:
        assert matrix.knows_user(user), \
            'expected original matrix user to be known in the matrix.'

    for user in matrix.get_users():
        assert user in algo_matrix_users, \
            'expected user to be present in the original matrix users.'


def assert_algo_matrix_user_rated_items(matrix: Matrix) -> None:
    """Assert the user rated items functionality of the specified matrix."""
    pytest.raises(KeyError, matrix.get_user_rated_items, algo_matrix_users.min() - 1)
    pytest.raises(KeyError, matrix.get_user_rated_items, algo_matrix_users.max() + 1)

    for user in matrix.get_users():
        rated_items = matrix.get_user_rated_items(user)

        assert len(rated_items) == len(pd.Series(rated_items).unique()), \
            'expected user rated items to be unique'

        for item in rated_items:
            assert item in algo_matrix_items, \
                'expected item to be present in the original matrix items.'


def create_algo_matrix(api_name: str, file_path: str=MATRIX_FILE) -> Matrix:
    """Create a matrix for the specified API name."""
    if api_name == IMPLICIT_API:
        return MatrixCSR(file_path)
    if api_name == LENSKIT_API:
        return Matrix(file_path)
    if api_name == SURPRISE_API:
        return MatrixSurprise(file_path, MATRIX_RATING_SCALE)

    raise NotImplementedError('unknown api matrix')


def test_algo_matrix() -> None:
    """Test if the algorithm matrix is sufficient to verify the other algorithm tests.

    The majority of the algorithms will pass with lower requirements. However, in particular
    the k-NN recommender algorithms require enough neighbours to produce the required amount
    of user-item recommendations.
    """
    assert len(algo_matrix_users) >= 5, \
        'expected more users for testing algorithms.'
    assert len(algo_matrix_items) >= 400, \
        'expected more items for testing algorithms.'


@pytest.mark.parametrize('api_name', [
    IMPLICIT_API, # MatrixCSR
    LENSKIT_API, # Matrix
    SURPRISE_API # MatrixSurprise
])
def test_algo_matrix_interface(api_name: str) -> None:
    """Test all functions of the matrix for the specified API name."""
    pytest.raises(FileNotFoundError, create_algo_matrix, api_name, '')

    matrix = create_algo_matrix(api_name)

    assert_algo_matrix_items(matrix)
    assert_algo_matrix_users(matrix)
    assert_algo_matrix_user_rated_items(matrix)
