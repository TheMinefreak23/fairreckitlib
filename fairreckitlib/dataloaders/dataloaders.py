
"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)

"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from utility import get_configs
from typing import Dict, Any
import os
from pandas.api.types import CategoricalDtype
from scipy import sparse
import zlib
import functools as ft

class dataloader_base(ABC):
    """_summary_
    """

    def __init__(self, config_path: str = "") -> None:
        if not config_path:
            config_path = os.path.dirname(os.path.abspath(__file__))
        self.configs = get_configs(os.path.join(config_path, "config.ini"))
        self._dataset_name = None
        self.data_frame = None
        self._user_cat = None
        self._item_cat = None
        self._filters = []
    
    @abstractmethod    
    def load_data(self) -> None:
        ...
    
    @abstractmethod    
    def filter_df(self) -> None:
        ...
    
    @abstractmethod    
    def get_user_item_matrix(self) -> None:
        ...

    def get_file_path(self, file_path: str, file_name: str) -> str:
        """
        """
        if not os.path.isdir(file_path):
            raise IOError('Failed to initialize DataRegistry: '
                          f'unknown data directory => {file_path}')
        dataset_path = os.path.join(file_path, file_name)
        if not os.path.isfile(dataset_path):
            raise IOError(f'Failed to find the dataset file: {dataset_path}')
        return dataset_path

    def get_info(self) -> Dict[str, Any]:
        """
        this function returns a dictionary containing some statistics about the data 
        this dictionary has key of string and value of any(e.g. int, float,bool)
        """
        return {
            'num_users': self.configs.getint(self._dataset_name, 'num_users'),
            'num_items': self.configs.getint(self._dataset_name, 'num_items'),
            'num_pairs': self.configs.getint(self._dataset_name, 'num_pairs'),
            'ratings': self.configs.get(self._dataset_name, 'ratings'),
            'min_rating': self.configs.getfloat(self._dataset_name, 'min_rating'),
            'max_rating': self.configs.getfloat(self._dataset_name, 'max_rating'),
            'timestamp': self.configs.getboolean(self._dataset_name, 'timestamp')
        }

    def save_to_tsv(self, file_path: str) -> None:
        """
        """
        self.data_frame.to_csv(file_path, header=False, sep='\t', index=False)

    def get_user_id(self, user_index: int) -> int:
        """
        this function takes the converted user index which is in user-item matrix and returns the user_ID in original dataset
        """
        df = pd.array([user_index])
        user_ids = list(df.astype(self._user_cat).cat.categories)
        return user_ids[0]

    def get_item_id(self, item_index: int) -> int:
        """
        this function takes item_index in user-item matrix and returns the item_ID in original dataset
        """
        df = pd.array([item_index])
        item_ids = list(df.astype(self._item_cat).cat.categories)
        return item_ids[0]


class dataloader_360k(dataloader_base):
    """
    In order to load 360k 
    """

    def __init__(self, config_path: str = "") -> None:
        super(dataloader_360k, self).__init__(config_path)
        self._dataset_name = "LFM-360K"
        self._filters = ["gender", "age", "country"]

    def load_data(self) -> None:
        """
        This function reads all the files of the dataset that its name is given,
        and loads the content of the files in dataframes based on the read config file.
        """
        params = dict(delimiter=self.configs.get("common", "DELIMITER", fallback=","),
                      names=self.configs.get(self._dataset_name, "headers").split(","),
                      engine='python', usecols=[0, 2, 3])
        self.data_frame = pd.read_csv(self.get_file_path(self.configs.get(self._dataset_name, 'file_path'),
                                                          self.configs.get(self._dataset_name, 'file_name')),
                                      **params)

    def filter_df(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        filters: a dictionary whose keys are the name of the column on which the filtering is being applied;
                 In case of age, the value must be a tuple, like (min_age, max_age)
        """
        file_name = "usersha1-profile.tsv"
        fields_map = {"gender": 1, "age": 2, "country": 3}
        use_cols = [0]
        use_cols.extend([fields_map[key] for key in filters])
        headers = ["user", "gender", "age", "country"]
        headers = [headers[i] for i in sorted(use_cols)]
        params = dict(delimiter=self.configs.get("common", "DELIMITER", fallback=","),
                      names=headers, engine='python', usecols=use_cols)
        df = pd.read_csv(self.get_file_path(self.configs.get(self._dataset_name, 'file_path'),
                                            file_name),
                         **params)
        values = {"gender": "", "age": -1, "Country": ""}
        df.fillna(value=values, inplace=True)
        df_filters = [df[k].map(lambda x: str(x).lower()) == str(v).lower() if k != 'age'
                      else df[k].astype(int).between(int(filters[k][0]), int(filters[k][1]), inclusive = "both")
                      for k, v in filters.items()]
        df = df[ft.reduce(lambda x, y: (x) & (y), df_filters)]["user"]
        return pd.merge(self.data_frame, df, on=["user"], how="left")
    
    def get_user_item_matrix(self, filters: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Void function to make sparse user-item matrix
        """
        if not self.data_frame:
            self.load_data()
            df = self.data_frame.copy()

        if filters:
            df = self.filter_df(filters)

        df["user_id"] = (df['user'].map(lambda x: zlib.adler32(str(x).encode('utf-8', errors='ignore'))))
        df["item_id"] = (df["item"].map(lambda x: zlib.adler32(str(x).encode('utf-8', errors='ignore'))))
        
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        shape = (len(users), len(items))

        # Create indices for users and items
        self._user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
        self._item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
        user_index = df["user_id"].astype(self._user_cat).cat.codes
        item_index = df["item_id"].astype(self._item_cat).cat.codes

        # Conversion via COO matrix
        csr = sparse.coo_matrix((df["rating"].astype(np.float32), (user_index, item_index)), shape=shape).tocsr()
        df = pd.DataFrame.sparse.from_spmatrix(csr)
        return df


def get_dataloader(dataset_name: str, config_path: str = "") -> dataloader_base:
    """
    this function takes the name of the dataset and returns its dataloader. the config_path is optional
    """
    if dataset_name.upper() == "LFM-360K":
        return dataloader_360k(config_path)


def explore_datasets(config_file_path: str, data_dir: str) -> None:
    """
    
    """
    if not os.path.isdir(data_dir):
        raise IOError('Failed to initialize DataRegistry: '
                      f'unknown data directory => {data_dir}')

    for file in os.listdir(data_dir):
        file_name = os.fsdecode(file)
        dataset_dir = os.path.join(data_dir, file_name)
        # skip all entries that are not a directory
        if not os.path.isdir(dataset_dir):
            continue

        dataset_path = os.path.join(dataset_dir, file_name + '.tsv')
        if not os.path.isfile(dataset_path):
            with open(dataset_path, 'a') as dfile:
                dfile.write('')

        import configparser
        config = configparser.RawConfigParser()
        if not config.has_section(file_name):
            config.add_section(file_name)
            config.set(file_name, 'file_path', dataset_dir)
            config.set(file_name, 'file_name', '')
            config.set(file_name, 'num_users', 0)
            config.set(file_name, 'num_items', 0)
            config.set(file_name, 'num_pairs', 0)
            config.set(file_name, 'ratings', '')
            config.set(file_name, 'min_rating', 0.0)
            config.set(file_name, 'max_rating', 0.0)
            config.set(file_name, 'timestamp', False)

            # Writing our configuration file to 'example.cfg'
            with open(config_file_path, 'a') as configfile:
                config.write(configfile)

