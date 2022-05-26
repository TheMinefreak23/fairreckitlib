"""This module contains counting functionality.

Both APC and ALC can be used in the Kullback-Leibler formula,
for which the module is also in this package.
As of now, they don't specifically count the 'artist'
play and listen count, but just whatever is in the 'item' column.
If you pass a dataframe for which the 'item' column contains artists,
it should work correctly, otherwise it will require changes to work properly.

Functions:

    get_item_dict: return dict with unique items.
    calculate_apc: count the artist play count.
    calculate_alc: count the artist listener count.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Dict
import pandas as pd


def get_item_dict(dataframe: pd.DataFrame) -> Dict[int, float]:
    """Make a dict with all unique items from the dataframe and starting value 0.

    Args:
        dataframe: the dataframe from which the items are taken.

    Returns:
        a dictionary with unique items as keys and 0 as values.
    """
    result = {}
    for item in dataframe['item'].unique():
        result[item] = 0
    return result


def calculate_apc(dataframe: pd.DataFrame) -> Dict[int, float]:
    """Sum up the total artist play count (apc).

    Used in the Kullback-Leibler formula for converting ratings.

    Args:
        dataframe with an item and rating header.

    Returns:
        a dictionary with key:item, value:apc.
    """
    apc = get_item_dict(dataframe)

    for _, row in dataframe.iterrows():
        apc[row['item']] += row['rating']

    return apc


def calculate_alc(dataframe: pd.DataFrame) -> Dict[int, float]:
    """Sum up the total artist listener count (alc).

    Used in the Kullback-Leibler formula for converting ratings.

    Args:
        dataframe with an item and rating header.

    Returns:
        a dictionary with key:item, value:alc.
    """
    alc = get_item_dict(dataframe)

    for _, row in dataframe.iterrows():
        alc[row['item']] += 1

    return alc
