"""This module tests the dataframe numerical filter functionality.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
from numpy import int64
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.numerical_filter import NumericalFilter
from src.fairreckitlib.data.filter.categorical_filter import CategoricalFilter
from src.fairreckitlib.data.filter.count_filter import CountFilter

from src.fairreckitlib.data.set.dataset_registry import DataRegistry

dataset = DataRegistry('tests/datasets').get_set('ML-100K-Sample')
filter_kwargs = {'dataset': dataset, 'matrix_name': 'user-movie-rating'}

class TestFilterAge:
    """Create a filter object and a dummy data frame to test age filter."""

    df_source = DataFrame({"id": [1, 2, 3, 4, 5], "age": [24, 0, -1, 45, 102]})
    filter_obj = NumericalFilter('', {}, **filter_kwargs)

    def test_run_no_param(self):
        """Test run with no given min max."""
        df_expected = DataFrame({"id": [1, 2, 4, 5], "age": [24, 0, 45, 102]})
        df_result = self.filter_obj.filter(self.df_source, 'age')
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param(self):
        """Test run with given parameters."""
        df_expected = DataFrame({"id": [1, 4], "age": [24, 45]})
        df_result = self.filter_obj.filter(self.df_source, 'age', min_val=10, max_val=45)
        assert_frame_equal(df_result, df_expected)

    def test_run_with_float_int_param(self):
        """Test run with given parameters."""
        df_expected = DataFrame({"id": [1, 4], "age": [24, 45]})
        df_result = self.filter_obj.filter(self.df_source, 'age',
            min_val=int(10), max_val=float(45.3))
        assert_frame_equal(df_result, df_expected)

    def test_run_no_age_col(self):
        """Test a given dataframe with no age column."""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        df_result = self.filter_obj.filter(df_given, 'age')
        assert_frame_equal(df_result, self.filter_obj.__empty_df__(df_given))


class TestFilterGender:
    """Create a filter object and a dummy data frame to test gender filter."""

    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6, 7],
        "user_gender": ['f', 'm', 'f', 'm', ' ', None, None]})
    filter_obj = CategoricalFilter('', {}, **filter_kwargs)
    df_empty = filter_obj.__empty_df__(df_source)

    def test_run_no_filter_param(self):
        """Test run with no given parameter."""
        df_result = self.filter_obj.filter(self.df_source)
        assert_frame_equal(df_result, self.df_empty)

    def test_run_with_one_filter(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'user_gender', ['f'])
        df_expected = DataFrame({"id": [1, 3], "user_gender": ['f', 'f']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_multiple_filters(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'user_gender', ['f', 'm'])
        df_expected = DataFrame({"id": [1, 2, 3, 4], "user_gender": ['f', 'm', 'f', 'm']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_multiple_filters_not_included(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'user_gender', ['f', 'm', 's'])
        df_expected = DataFrame({"id": [1, 2, 3, 4], "user_gender": ['f', 'm', 'f', 'm']})
        assert_frame_equal(df_result, df_expected)

    def test_run_no_gender_col(self):
        """Test a given dataframe with no gender column."""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        df_result = self.filter_obj.filter(df_given, 'gender', ['m'])
        assert_frame_equal(df_result, self.filter_obj.__empty_df__(df_given))

    def test_run_with_none(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'user_gender', [None])
        df_expected = DataFrame({"id": [6, 7], "user_gender": [None, None]})
        assert_frame_equal(df_result, df_expected)


class TestFilterCountry:
    """Create a filter object and a dummy data frame to test country filter."""

    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6, 7],
                           "country": ['Iran', ' ', 'Netherlands', 'Russia', 'Sweden', None, None]})
    filter_obj = CategoricalFilter('', {}, **filter_kwargs)
    empty_df = filter_obj.__empty_df__(df_source)

    def test_run_no_param(self):
        """Test run with no given parameter."""
        df_result = self.filter_obj.filter(self.df_source)
        assert_frame_equal(df_result, self.empty_df)

    def test_run_no_conditions_param(self):
        """Test run with no given parameter."""
        df_result = self.filter_obj.filter(self.df_source, 'country')
        assert_frame_equal(df_result, self.empty_df)

    def test_run_no_found_param(self):
        """Test run with no found parameter."""
        df_result = self.filter_obj.filter(self.df_source, 'country', ['Belgium'])
        df_expected = self.empty_df
        df_expected.id = df_expected.id.astype(int64)
        df_expected.country = df_expected.country.astype(str)
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'country', ['Iran'])
        df_expected = DataFrame({"id": [1], "country": ['Iran']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_multiple_matching_filter(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'country', ['Iran', 'Sweden'])
        df_expected = DataFrame({"id": [1, 5], "country": ['Iran', 'Sweden']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_multiple_unmatching_filter(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'country',
            ['Iran', 'Sweden', 'France', 'Hungary'])
        df_expected = DataFrame({"id": [1, 5], "country": ['Iran', 'Sweden']})
        assert_frame_equal(df_result, df_expected)

    def test_run_no_country_col(self):
        """Test a given dataframe with no country column."""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        df_result = self.filter_obj.filter(df_given, 'country', ['Sweden'])
        assert_frame_equal(df_result, self.filter_obj.__empty_df__(df_given))

    def test_run_with_none(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'country', [None])
        df_expected = DataFrame({"id": [6, 7], "country": [None, None]})
        assert_frame_equal(df_result, df_expected)


class TestFilterCount:
    """Create a filter object and a dummy data frame to test count filter."""

    df_source = DataFrame({"id":[1, 2, 3, 4, 5, 6, 7, 8, 9],
                           "country": ["NL", "NL", "NL", "BE", "NL", "BE", "FR", None, None]})
    filter_obj = CountFilter('', {}, **filter_kwargs)
    df_empty = filter_obj.__empty_df__(df_source)

    def test_run_not_exist_col(self):
        """Test run with non-existent column."""
        df_result = self.filter_obj.filter(self.df_source, "age")
        assert_frame_equal(df_result, self.df_empty)

    def test_run_with_param(self):
        """Test run with existent column."""
        df_result = self.filter_obj.filter(self.df_source, "country", 2)
        df_expected = DataFrame({"id":[1, 2, 3, 4, 5, 6, 8, 9],
                                 "country": ["NL", "NL", "NL", "BE", "NL", "BE", None, None]})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_low_threshold(self):
        """Test run with threshold < 1."""
        df_result = self.filter_obj.filter(self.df_source, "country", -5)
        assert_frame_equal(df_result, self.df_source)

    def test_run_with_high_threshold(self):
        """Test run with infinite threshold."""
        df_result = self.filter_obj.filter(self.df_source, "country", math.inf)
        assert_frame_equal(df_result, self.df_empty)
