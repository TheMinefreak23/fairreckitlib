"""This module tests the dataframe count filter functionality.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.count_filter import CountFilter
from src.fairreckitlib.data.set.dataset_registry import DataRegistry

dataset = DataRegistry('tests/datasets').get_set('ML-100K-Sample')
filter_kwargs = {'dataset': dataset, 'matrix_name': 'user-movie-rating'}

class TestFilterCount:
    """Creates a filter object and a dummy data frame to test count filter."""

    df_source = DataFrame({"id":[1, 2, 3, 4, 5, 6, 7],
                           "country": ["NL", "NL", "NL", "BE", "NL", "BE", "FR"]})
    filter_obj = CountFilter('', {}, **filter_kwargs)
    df_empty = filter_obj.__empty_df__(df_source)

    def test_run_not_exist_col(self):
        """Test run with non-existent column."""
        df_result = self.filter_obj.filter(self.df_source, "age")
        assert_frame_equal(df_result, self.df_empty)

    def test_run_with_param(self):
        """Test run with existent column."""
        df_result = self.filter_obj.filter(self.df_source, "country", 2)
        df_expected = DataFrame({"id":[1, 2, 3, 4, 5, 6],
                                 "country": ["NL", "NL", "NL", "BE", "NL", "BE"]})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_low_threshold(self):
        """Test run with threshold < 1."""
        df_result = self.filter_obj.filter(self.df_source, "country", -5)
        assert_frame_equal(df_result, self.df_source)


    def test_run_with_high_threshold(self):
        """Test run with infinite threshold."""
        df_result = self.filter_obj.filter(self.df_source, "country", math.inf)
        assert_frame_equal(df_result, self.df_empty)
