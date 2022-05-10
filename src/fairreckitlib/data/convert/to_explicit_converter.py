"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base_converter import RatingConverter

class ToExplicitConverter(RatingConverter):
    """Class with the functionality to convert ratings to an explicit rating between [0,1]"""
    def __init__(self):
        pass

    def run(self, df):
        """Converts ratings in the df to explicit values.
        Takes the max value and divides all values so
        they all fall within a range of [0,1].

        Args:
            df(pandas.DataFrame): a df that should contain a 'rating' header.
        
        Returns:
            df(pandas.DataFrame): with the converted rating values.
        """
        max_rating = df.max()['rating']
        df['rating'] = df['rating'].apply(lambda x : round(x / max_rating))
        return df

def create_to_explicit_converter():
    return ToExplicitConverter()
