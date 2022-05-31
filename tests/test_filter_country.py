"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from numpy import int64
from src.fairreckitlib.data.filter.country import CountryFilter


class TestFilterCountry:
    """Create a filter object and a dummy data frame to test country filter."""

    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6],
                           "country": ['Iran', ' ', 'Netherlands', 'Russia', 'Sweden', None]})

    def test_run_no_param(self):
        """Test run with no given parameter."""

        filter_obj = CountryFilter(self.df_source)
        df_result = filter_obj.run("country")
        assert_frame_equal(df_result, self.df_source)

    def test_run_no_found_param(self):
        """Test run with no found parameter."""

        filter_obj = CountryFilter(self.df_source, ['Belgium'])
        df_result = filter_obj.run("country")
        df_expected = DataFrame({"id": [], "country": []})
        df_expected.id = df_expected.id.astype(int64)
        df_expected.country = df_expected.country.astype(str)
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param(self):
        """Test run with given parameters."""

        filter_obj = CountryFilter(self.df_source, ['Iran'])
        df_result = filter_obj.run("country")
        df_expected = DataFrame({"id": [1], "country": ['Iran']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param_multi_filter_1(self):
        """Test run with given parameters."""

        filter_obj = CountryFilter(self.df_source, ['Iran', 'Belgium'])
        df_result = filter_obj.run("country")
        df_expected = DataFrame({"id": [1], "country": ['Iran']})
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param_multi_filter_2(self):
        """Test run with given parameters."""

        filter_obj = CountryFilter(self.df_source, ['Iran', 'Sweden'])
        df_result = filter_obj.run("country")
        df_expected = DataFrame({"id": [1, 5], "country": ['Iran', 'Sweden']})
        assert_frame_equal(df_result, df_expected)


def test_run_no_country():
    """Test a given dataframe with no country column."""

    df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
    filter_obj = CountryFilter(df_given, ['Sweden'])
    df_result = filter_obj.run("country")
    assert_frame_equal(df_result, df_given)
