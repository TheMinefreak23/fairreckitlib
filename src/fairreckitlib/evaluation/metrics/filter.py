"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import enum


class Filter(enum.Enum):
    """
    Data filter/selection
    """
    EQUALS = 'equals' # Select equal rows (e.g. selecting a category lke males)
    CLAMP = 'clamp' # Select rows with values between a min and max


def filter_data(dataframe, filters):
    """
    Filter a dataset using the specified list of filter passes

    :param dataframe: The dataset (Pandas dataframe) to be filtered
    :param filters: The filters used
    :return: The filtered dataframe
    """
    for filter_pass in filters:
        name = filter_pass['name']
        value = filter_pass['value']
        #print(filter_pass)
        #print(filter_pass['type'])
        if filter_pass['type'] == Filter.EQUALS.value:
            # Rows that are equal to the
            condition = dataframe[name] == value
        elif filter_pass['type'] == Filter.CLAMP.value:
            # Exclusive max
            condition = value['min'] <= dataframe[name] < value['max']
        else:
            raise Exception # Type of filter_pass not found

        dataframe = dataframe[condition]
    return dataframe
