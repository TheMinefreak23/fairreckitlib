"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
import lenskit.crossfold as xf

from fairreckitlib.data.split import factory, random, temporal

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers and indices
# [user, artistid, artistname, plays]
df_lfm360k_sample = pd.read_csv(
    'tests\datasets\sample\lfm-360k-sample.tsv', delimiter='\t')

# sample of the first 1000 entries of the ml-100k dataset
# this already has headers and indices
# [user, item, rating, timestamp]
df_ml100k_sample = pd.read_csv(
    'tests\datasets\sample\ml-100k-sample.tsv', delimiter='\t')

# the list of dataframes to test splitting with
dfs =  [('df_lfm360k', df_lfm360k_sample),
        ('df_ml100k' , df_ml100k_sample)]

# creating the factories to run splitting with
split_factory = factory.create_split_factory()
random_split = split_factory.create('random')
temp_split = split_factory.create('temporal')


# the list of test ratios to test splitting with 
# should be a 0.0 < float < 1.0
ratios = [0.2, 0.3, 0.8]



# tests if the created variables are in fact of that class
def test_split_classes():
    assert type(split_factory) is factory.SplitFactory
    assert type(random_split) is random.RandomSplitter
    assert type(temp_split) is temporal.TemporalSplitter

# tests if the temporal split returns a tuple with the test set being
# the size of the ratio, with a 75% margin
# larger margin because it is split on user timestamps, which differs slightly
# compared to the overall timestamps in the dataset
def test_temp_split():
    for (df_name, df) in dfs:
        if 'timestamp' in df: # this can be hcanged to 'assert', if you want it to fail instead of skip timestamp-less dataframes
            for test_ratio in ratios:
                (train, test) = temp_split.run(df, test_ratio)
                assert len(train.index) != 0, 'Train set is empty: ' + df_name + str(test_ratio) 
                assert len(test.index) != 0, 'Test set is empty: ' + df_name + str(test_ratio)

                ratio = len(test.index) / (len(train.index) + len(test.index))
                assert ratio < (test_ratio * 1.75) and ratio > (test_ratio * 0.25), 'Test set should be around ' + str(test_ratio) + ': ' + df_name

                # for every row, assert that the timestamps are bigger in the test set per user
                for index, row in train.iterrows():
                    for index_, row_ in test.iterrows():
                        if row['user'] == row_['user']:
                            assert row['timestamp'] <= row_['timestamp']

# tests if the random split returns a tuple with the test set being
# the size of the ratio, with a 10% margin
def test_random_split():
    for (df_name, df) in dfs:
        for test_ratio in ratios:
            (train, test) = random_split.run(df, test_ratio)
            assert len(train.index) != 0, 'Train set is empty: ' + df_name + str(test_ratio) 
            assert len(test.index) != 0, 'Test set is empty: ' + df_name + str(test_ratio)

            ratio = len(test.index) / (len(train.index) + len(test.index))
            assert ratio < (test_ratio * 1.1) and ratio > (test_ratio * 0.9), 'Test set should be around ' + str(test_ratio) + ': ' + df_name

