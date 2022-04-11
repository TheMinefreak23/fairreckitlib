""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import enum


class Filter(enum.Enum):
    Equals = 'equals' # Select equal rows (e.g. selecting a category lke males)
    Clamp = 'clamp' # Select rows with values between a min and max


def filter(df, filters):
    for filter in filters:
        name = filter['name']
        value = filter['value']
        print(filter)
        print(filter['type'])
        if filter['type'] == Filter.Equals.value:
            # Rows that are equal to the
            condition = df[name] == value
        elif filter['type'] == Filter.Clamp.value:
            # Exclusive max
            condition = value['min'] <= df[name] < value['max']
        else:
            raise Exception # Type of filter not found

        df = df[condition]
    return df
