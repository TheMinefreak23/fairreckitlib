"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base import DataLoader
from dataloaders import dataloaders as dl


class DataLoaders(DataLoader):
    """
    _summary_
    """
    def __init__(self) -> None:
        super().__init__()

    def run(self, dataset_name: str, config_path: str = "") -> dl.DataloaderBase:
        """
        _summary_
        """
        return dl.get_dataloader(dataset_name, config_path)
