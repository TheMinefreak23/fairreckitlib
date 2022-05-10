"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

def get_item_dict(df):
    """Makes a dict with all unique items from the df."""
    result = {}
    for item in df['item'].unique():
        result[item] = 0
    return result

"""
APC/ALC are used in the Kullback-Leibler formula for converting
implicit ratings to explicit ratings.
As of now, a fake implementation is used instead of the KB formula.
"""

def calculate_apc(df):
    """Sums up the total artist play count (apc).

    Args:
        df(pandas.DataFrame) with an item and rating header.
    Returns:
        dict with key:item, value:apc
    """
    apc = get_item_dict(df)

    for _, row in df.iterrows():
        apc[row['item']] += row['rating']

    return apc

def calculate_alc(df):
    """Sums up the total artist listener count (alc).

    Args:
        df(pandas.DataFrame) with an item and rating header.
    Returns:
        dict with key:item, value:alc
    """
    alc = get_item_dict(df)

    for _, row in df.iterrows():
        alc[row['item']] += 1

    return alc
