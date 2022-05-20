"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.gender import GenderFilter
<<<<<<< HEAD
=======

>>>>>>> 911fb2e3a9e9a3014c8737e3368d9b7f3cb478cd

class TestFilterGender:
    """Create a filter object and a dummy data frame to test gender filter."""
    df_source = DataFrame({"id": [1, 2, 3, 4, 5, 6], "gender": ['f', 'm', 'f', 'm', ' ', None]})
    filter_obj = GenderFilter(df_source)

    def test_run_no_param(self):
        """Test run with no given parameter."""
        df_result = self.filter_obj.run('')
        assert_frame_equal(df_result, self.df_source)

    def test_run_with_param(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.run('FEmale')
        df_expected = DataFrame({"id": [1, 3], "gender": ['f', 'f']})
        assert_frame_equal(df_result, df_expected)
    @classmethod
    def test_run_no_gender(cls):
        """Test a given dataframe with no gender column."""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        filter_obj = GenderFilter(df_given)
        df_result = filter_obj.run('m')
        assert_frame_equal(df_result, df_given)
