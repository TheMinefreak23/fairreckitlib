"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from fairreckitlib.data.split import factory, random, temporal

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers and indices
# [user, artistid, artistname, plays]
df_lfm360k_sample = pd.read_csv(
    'tests\\datasets\\sample\\lfm-360k-sample.tsv', delimiter='\t')

# sample of the first 1000 entries of the ml-100k dataset
# this already has headers and indices
# [user, item, rating, timestamp]
df_ml100k_sample = pd.read_csv(
    'tests\\datasets\\sample\\ml-100k-sample.tsv', delimiter='\t')

# the list of dataframes to test splitting with
dfs =  [('df_lfm360k', df_lfm360k_sample),
        ('df_ml100k' , df_ml100k_sample)]

# creating the factories to run splitting with
split_factory = factory.create_split_factory()
random_split = split_factory.create('random', dict())
temp_split = split_factory.create('temporal', dict())


# the list of test ratios to test splitting with
# should be a 0.0 < float < 1.0
ratios = [0.2, 0.3, 0.8]




def test_split_classes():
    """tests if the created variables are in fact of that class"""
    assert isinstance(split_factory, factory.SplitFactory)
    assert isinstance(random_split, random.RandomSplitter)
    assert isinstance(temp_split, temporal.TemporalSplitter)


def test_temp_split():
    """ tests if the temporal split returns a tuple with the test set being
    the size of the ratio, with a 75% margin
    larger margin because it is split on user timestamps, which differs slightly
    compared to the overall timestamps in the dataset
    """
    for (df_name, df) in dfs:
        if 'timestamp' in df:
            for test_ratio in ratios:
                (train, test) = temp_split.run(df, test_ratio)
                assert len(train.index) != 0, \
                    'Train set is empty: ' + df_name + str(test_ratio)
                assert len(test.index) != 0, \
                    'Test set is empty: ' + df_name + str(test_ratio)

                ratio = len(test.index) / (len(train.index) + len(test.index))
                assert (test_ratio * 0.25) < ratio < (test_ratio * 1.75), \
                    'Test set should be around ' + str(test_ratio) + ': ' + df_name

                # for every row, assert that the timestamps are bigger in the test set per user
                for _, row in train.iterrows():
                    for _, row_ in test.iterrows():
                        if row['user'] == row_['user']:
                            assert row['timestamp'] <= row_['timestamp']


def test_random_split():
    """tests if the random split returns a tuple with the test set being
    the size of the ratio, with a 10% margin
    """
    for (df_name, df) in dfs:
        for test_ratio in ratios:
            (train, test) = random_split.run(df, test_ratio)
            assert len(train.index) != 0, 'Train set is empty: ' + df_name + str(test_ratio)
            assert len(test.index) != 0, 'Test set is empty: ' + df_name + str(test_ratio)

            ratio = len(test.index) / (len(train.index) + len(test.index))
            assert (test_ratio * 0.9) < ratio < (test_ratio * 1.1), \
                'Test set should be around ' + str(test_ratio) + ': ' + df_name
