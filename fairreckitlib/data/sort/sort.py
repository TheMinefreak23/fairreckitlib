"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas

def sort_data(df, header, ascending):
    """Returns the dataset, sorted by the given header, ascending or descending."""
    return df.sort_values(by=header, ascending=ascending)
