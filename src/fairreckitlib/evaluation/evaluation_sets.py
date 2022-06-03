"""This module contains evaluation sets file paths and dataframes definition.

Classes:

    EvaluationSetPaths: the file paths of the evaluation sets.
    EvaluationSets: the dataframes of the evaluation sets.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class EvaluationSetPaths:
    """Evaluation set file paths.

    train_path: the train set file path.
    test_path: the test set file path.
    ratings_path: the computed rating set file path.
    """

    train_path: str
    test_path: str
    ratings_path: str


@dataclass
class EvaluationSets:
    """Evaluation set dataframes.

    train: the train set or None when not needed for evaluation.
    test: the test set or None when not needed for evaluation.
    ratings: the computed ratings set.
    """

    train: Optional[pd.DataFrame]
    test: Optional[pd.DataFrame]
    ratings: pd.DataFrame
