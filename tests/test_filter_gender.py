"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.categorical_filter import CategoricalFilter


class TestFilterGender:
    """Create a filter object and a dummy data frame to test gender filter."""
    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6, 7], "gender": ['f', 'm', 'f', 'm', ' ', 'n', None]})
    filter_obj = CategoricalFilter()

    def test_run_no_param(self):
        """Test run with no given parameter."""
        df_result = self.filter_obj.filter(self.df_source)
        assert_frame_equal(df_result, self.df_source)

    def test_run_with_param(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'gender', ['f'])
        df_expected = DataFrame({"id": [1, 3], "gender": ['f', 'f']})
        assert_frame_equal(df_result, df_expected)
        
    def test_run_with_multiple_param(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.filter(self.df_source, 'gender', ['f', 'm'])
        df_expected = DataFrame({"id": [1, 2, 3, 4], "gender": ['f', 'm', 'f', 'm']})
        assert_frame_equal(df_result, df_expected)

def test_run_no_gender_col():
    """Test a given dataframe with no gender column."""
    df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
    filter_obj = CategoricalFilter()
    df_result = filter_obj.filter(df_given, 'gender', ['m'])
    assert_frame_equal(df_result, df_given)
