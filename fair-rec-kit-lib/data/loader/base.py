"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


"""
base class to preprocess a dataset to a standardized interface:

Filename: dataset_dir.tsv
Headers: False
Index: False
Columns: user item rating (timestamp)

'user': int
'item': int
'rating': int/float
'timestamp': timestamp

Creating additional artifacts/files is fine as long as this is included in the metadata.
Some extra metadata needs to be generated about which other files are present
and how they are linked to the 'user' and/or 'item' columns.
This metadata is needed in the Dataset class.

Together with a factory pattern similar to the data.split module
we can define a variety of loaders for all of the datasets.

Moreover this means that a future dev can easily add a new dataset.
"""
class DataLoader(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def run(self, dataset_dir, **kwargs):
        raise NotImplementedError()
