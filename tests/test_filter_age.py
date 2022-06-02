"""This module tests the dataframe numerical filter functionality.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from logging.handlers import DEFAULT_SOAP_LOGGING_PORT
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.numerical_filter import NumericalFilter
# TODO use sample datasets
class TestFilterAge:
    """Creates a filter object and a dummy data frame to test age filter."""

    df_source = DataFrame({"id": [1, 2, 3, 4, 5], "age": [24, 0, -1, 45, 102]})
    filter_obj = NumericalFilter()

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
        df_result = self.filter_obj.filter(self.df_source, 'age', min_val=int(10), max_val=float(45.3))
        assert_frame_equal(df_result, df_expected)
    
    def test_run_no_age_col(self):
        """Test a given dataframe with no age column."""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        df_result = self.filter_obj.filter(df_given, 'age')
        assert_frame_equal(df_result, self.filter_obj.__empty_df__(df_given))
