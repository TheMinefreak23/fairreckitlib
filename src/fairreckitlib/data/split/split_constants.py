"""This module contains splitting constants that are used in other modules.

Constants:

    KEY_SPLITTING: key that is used to identify splitters.
    KEY_SPLIT_TEST_RATIO: key that is used to identify the splitter test ratio.
    SPLIT_RANDOM: name of the random splitter.
    SPLIT_TEMPORAL: name of the temporal splitter.
    DEFAULT_SPLIT_NAME: the name of the default splitter.
    DEFAULT_SPLIT_TEST_RATIO: the default split test ratio.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

SPLIT_RANDOM = 'random'
SPLIT_TEMPORAL = 'temporal'

KEY_SPLITTING = 'splitting'
KEY_SPLIT_TEST_RATIO = 'test_ratio'

DEFAULT_SPLIT_NAME = SPLIT_RANDOM
DEFAULT_SPLIT_TEST_RATIO = 0.2
