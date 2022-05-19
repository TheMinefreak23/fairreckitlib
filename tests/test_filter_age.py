"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.data.filter.age import AgeFilter
from pandas.testing import assert_frame_equal
from pandas import DataFrame

class TestFilterAge:
    """Creates a filter object and a dummy data frame to test age filter."""
    df_source = DataFrame({"id": [1, 2, 3, 4, 5], "age": [24, 0, -1, 45, 102]})
    filter_obj = AgeFilter(df_source)

    def test_run_no_param(self):
        """Test run with no given parameter."""
        df_expected = DataFrame({"id": [1, 2, 4], "age": [24, 0, 45]})
        df_result = self.filter_obj.run()
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param(self):
        """Test run with given parameters."""
        df_result = self.filter_obj.run(min_val=10, max_val=45)
        df_expected = DataFrame({"id": [1, 4], "age": [24, 45]})
        assert_frame_equal(df_result, df_expected)

    def test_run_no_age(self):
        """Test a given dataframe with no age column."""
        df_given = DataFrame({"id": [1, 2, 3, 4, 5], "play_count": [24, 0, -1, 45, 102]})
        filter_obj = AgeFilter(df_given)
        df_result = filter_obj.run()
        assert_frame_equal(df_result, df_given)
