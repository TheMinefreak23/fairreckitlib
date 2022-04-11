
"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)

"""
from abc import ABC
from pyclbr import Class
import pandas as pd
import numpy as np
from utility import get_configs
from typing import List
import os
from pandas.api.types import CategoricalDtype
from scipy import sparse
import zlib

class dataloader_base(ABC):
    """_summary_
    """
    configs = None
    _dataset_name = None
    data_frames = {}
    _user_cat = None
    _item_cat = None

    def __init__(self) -> None:
        self.configs = get_configs(os.path.dirname(os.path.abspath(__file__)) + "\\config.ini")

class dataloader_360k(dataloader_base):
    """_summary_
    """
    
    def __init__(self):
        super(dataloader_360k, self).__init__()
        self._dataset_name = "lfm_360k"

    def load_data(self) -> None:
        """
        This function reads all the files of the dataset that its name is given,
        and loads the content of the files in dataframes based on the read config file.
        """

        for sub_dataset in [section for section in self.configs.sections() if section.startswith(self._dataset_name)]:
            params = dict(delimiter=self.configs.get(sub_dataset, "delimiter", fallback=","),
                          names=self.configs.get(sub_dataset, "headers").split(","),
                          engine='python')
            if self.configs.get(sub_dataset, "timestamp", fallback=None):
                params.update(dict(parse_dates=self.configs.get(sub_dataset, "timestamp").split(",")))
            df = pd.read_csv(self.configs.get(sub_dataset, "file_path"), **params)
            self.data_frames.update({sub_dataset: df})
    
    def get_user_item_matrix(self) -> sparse.csr_matrix:
        """
        """
        df = self.data_frames["lfm_360k_usersha1-artmbid-artname-plays"].copy()

        df['user_id'] = df['user-mboxsha1'].map(lambda x: zlib.adler32(str(x).encode('utf-8', errors='ignore')))
        df['artist_id'] = df['artist-name'].map(lambda x: zlib.adler32(str(x).encode('utf-8', errors='ignore')))
        
        self.data_frames["lfm_360k_usersha1-artmbid-artname-plays"] = df

        users = df["user_id"].unique()
        items = df["artist_id"].unique()
        shape = (len(users), len(items))

        # Create indices for users and items
        self._user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
        self._item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
        user_index = df["user_id"].astype(self._user_cat).cat.codes
        item_index = df["artist_id"].astype(self._item_cat).cat.codes

        # Conversion via COO matrix
        coo = sparse.coo_matrix((df["plays"].astype(np.float32), (user_index, item_index)), shape=shape)
        return coo.tocsr()


    def get_user_id(self, user_index: int) -> int:
        """
        """
        df = pd.array([user_index])
        user_ids = list(df.astype(self._user_cat).cat.categories)
        return user_ids[0]

    def get_item_id(self, item_index) -> List[int]:
        """
        """
        df = pd.array([item_index])
        item_ids = list(df.astype(self._item_cat).cat.categories)
        return item_ids[0]

def get_dataloader(dataset_name: str) -> dataloader_base:
# def dataloader(dataset_name: str) -> dataloader_base:
    """
    """
    if dataset_name == "lfm_360k":
        return dataloader_360k()
