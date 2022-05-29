"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.gender import GenderFilter


class TestFilterGender:
    """Create a filter object and a dummy data frame to test gender filter."""
    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6],
                           "user_gender": ['f', 'm', 'f', 'm', ' ', None]})
    filter_obj = GenderFilter(df_source)

    def test_run_no_filter_param(self):
        """Test run with no given parameter."""
        df_result = self.filter_obj.run("user_gender", [])
        assert_frame_equal(df_result, self.df_source)

    def test_run_with_one_filter(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.run("user_gender", ['FEmale'])
        df_expected = DataFrame({"id": [1, 3], "user_gender": ['f', 'f']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_multi_filters(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.run("user_gender", ['FEmale', 'M'])
        df_expected = DataFrame({"id": [1, 2, 3, 4], "user_gender": ['f', 'm', 'f', 'm']})
        assert_frame_equal(df_result, df_expected)


def test_run_no_gender():
    """Test a given dataframe with no gender column."""
    df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
    filter_obj = GenderFilter(df_given)
    df_result = filter_obj.run("user_gender", 'm')
    assert_frame_equal(df_result, df_given)
