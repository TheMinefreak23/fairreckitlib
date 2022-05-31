"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.count import CountFilter

class TestFilterCount:
    """Creates a filter object and a dummy data frame to test count filter."""
    df_source = DataFrame({"id": [1, 2, 3, 4, 5], "count": [24, 0, -1, 45, 102]})

    def test_run_no_param(self):
        """Test run with no given parameter."""
        filter_obj = CountFilter(self.df_source)
        df_expected = DataFrame({"id": [1, 2, 4, 5], "count": [24, 0, 45, 102]})
        df_result = filter_obj.run("count")
        assert_frame_equal(df_result, df_expected)

    def test_run_with_param(self):
        """Test run with given parameters."""
        filter_obj = CountFilter(self.df_source, min_val=10, max_val=45)
        df_result = filter_obj.run("count")
        df_expected = DataFrame({"id": [1, 4], "count": [24, 45]})
        assert_frame_equal(df_result, df_expected)


def test_run_no_count():
    """Test a given dataframe with no count column."""
    df_given = DataFrame({"id": [1, 2, 3, 4, 5], "age": [24, 0, -1, 45, 102]})
    filter_obj = CountFilter(df_given)
    df_result = filter_obj.run("count")
    assert_frame_equal(df_result, df_given)
