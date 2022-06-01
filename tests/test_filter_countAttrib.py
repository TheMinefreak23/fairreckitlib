"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.countAttrib import CountAttribFilter

class TestFilterCountAttrib:
    """Creates a filter object and a dummy data frame to test countAttrib filter."""

    df_source = DataFrame({"country": ["NL", "NL", "NL", "BE", "NL", "BE", "FR"],
                           "id":[4000, 7032, 39, 5, 8, 9003, 1]})
    filter_obj = CountAttribFilter(df_source)

    def test_run_not_exist_col(self):
        """Test run with non-existent column."""
        df_result = self.filter_obj.run("age", 2)
        assert_frame_equal(df_result, self.df_source)

    def test_run_with_param(self):
        """Test run with existent column."""
        df_result = self.filter_obj.run("country", 2)
        df_expected = DataFrame({"country": ["NL", "NL", "NL", "BE", "NL", "BE"],
                                 "id":[4000, 7032, 39, 5, 8, 9003]})
        assert_frame_equal(df_result, df_expected)
