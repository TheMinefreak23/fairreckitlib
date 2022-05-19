"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.data.filter.country import CountryFilter
from pandas.testing import assert_frame_equal
from pandas import DataFrame
from numpy import int64

class TestFilterCountry:
    """Create a filter object and a dummy data frame to test country filter"""
    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6],
                           "country": ['Iran', ' ', 'Netherlands', 'Russia', 'Sweden', None]})
    filter_obj = CountryFilter(df_source)

    def test_run_no_param(self):
        """Test run with no given parameter"""
        df_result = self.filter_obj.run()
        df_expected = DataFrame({"id": [], "country": []})
        df_expected.id = df_expected.id.astype(int64)
        df_expected.country = df_expected.country.astype(str)
        assert_frame_equal(df_result, df_expected)
        
    def test_run_no_param(self):
        """Test run with nof fount parameter"""
        df_result = self.filter_obj.run('Belgium')
        df_expected = DataFrame({"id": [], "country": []})
        df_expected.id = df_expected.id.astype(int64)
        df_expected.country = df_expected.country.astype(str)
        assert_frame_equal(df_result, df_expected)
        
    def test_run_with_param(self):
        """Test run with given parameters"""
        df_result = self.filter_obj.run('Iran')
        df_expected = DataFrame({"id": [1], "country": ['Iran']})
        assert_frame_equal(df_result, df_expected)

    def test_run_no_country(self):
        """Test a given dataframe with no country column"""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        filter_obj = CountryFilter(df_given)
        df_result = filter_obj.run('Sweden')
        assert_frame_equal(df_result, df_given)