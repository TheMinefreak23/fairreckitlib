
"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)

"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
import zlib
import functools as ft
import h5py
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy import sparse
from .utility import get_configs

class DataloaderBase(ABC):
    """
    Abstract class
    """

    def __init__(self, config_path: str = "") -> None:
        if not config_path:
            config_path = os.path.dirname(os.path.abspath(__file__))
        self.configs = get_configs(os.path.join(config_path, "config.ini"))
        self._dataset_name = None
        self.ui_data_frame = pd.DataFrame([])
        self._user_cat = None
        self._item_cat = None
        self._filter_options = []

    @abstractmethod
    def load_data(self) -> None:
        """
        the abstract method obliges the developer to overwrite the method
        """
        raise NotImplementedError()

    @abstractmethod
    def filter_df(self, filters: Dict[str, Any]) -> None:
        """
        abstract method of the class
        """
        raise NotImplementedError()

    @abstractmethod
    def df_add_column(self, add_columns: List[str]) -> None:
        """
        abstract method of the class to add columns
        """
        raise NotImplementedError()

    def df_remove_column(self, column_list: List[str]) -> None:
        """
        general method of the class to remove columns
        """
        headers = self.ui_data_frame.columns
        common_elements = np.intersect1d(column_list, headers)
        if not common_elements:
            return
        self.ui_data_frame.drop(columns=common_elements, inplace=True)

    def df_formatter(self, columns_add: List[str] = None, columns_remove: List[str] = None) -> None:
        """
        general method of the class to add/remove columns and returns the updated data frame
        """
        if columns_add:
            self.df_add_column(columns_add)
        if columns_remove:
            self.df_remove_column(columns_remove)

    def get_user_item_matrix(self) -> sparse.csr:
        """
        Void function to make sparse user-item matrix
        """
        if not self.ui_data_frame:
            self.load_data()

        data_frame = self.ui_data_frame
        data_frame["user_id"] = (data_frame['user']
                                     .map(lambda x: zlib.adler32(str(x).encode('utf-8',
                                                                               errors='ignore'))))
        data_frame["item_id"] = (data_frame["item"]
                                     .map(lambda x: zlib.adler32(str(x).encode('utf-8',
                                                                               errors='ignore'))))

        users = data_frame["user_id"].unique()
        items = data_frame["item_id"].unique()
        shape = (len(users), len(items))

        # Create indices for users and items
        self._user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
        self._item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
        user_index = data_frame["user_id"].astype(self._user_cat).cat.codes
        item_index = data_frame["item_id"].astype(self._item_cat).cat.codes

        # Conversion via COO matrix
        csr = sparse.coo_matrix((data_frame["rating"].astype(np.float32), (user_index, item_index)),
                                shape=shape).tocsr()
        return csr

    @classmethod
    def get_file_path(cls, file_path: str, file_name: str) -> str:
        """
        this function takes the class and file path and file name to make a string of file path
        where dataset exists
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
        this function takes the file path and saves tsv version of the data frame
        """
        self.ui_data_frame.to_csv(file_path, header=False, sep='\t', index=False)

    def get_user_id(self, user_index: int) -> int:
        """
        this function takes the converted user index which is in user-item matrix and
        returns the user_ID in original dataset
        """
        data_frame = pd.array([user_index])
        user_ids = list(data_frame.astype(self._user_cat).cat.categories)
        return user_ids[0]

    def get_item_id(self, item_index: int) -> int:
        """
        this function takes item_index in user-item matrix and
        returns the item_ID in original dataset
        """
        data_frame = pd.array([item_index])
        item_ids = list(data_frame.astype(self._item_cat).cat.categories)
        return item_ids[0]


class Dataloader360K(DataloaderBase):
    """
    In order to load 360k
    """

    def __init__(self, config_path: str = "") -> None:
        super().__init__(config_path)
        self._dataset_name = "LFM-360K"
        self._filter_options = ["user", "item", "rating", "gender", "age", "country"]

    def load_data(self) -> None:
        """
        This function reads all the files of the dataset that its name is given,
        and loads the content of the files in dataframes based on the read config file.
        """
        params = dict(delimiter=self.configs.get("common", "DELIMITER", fallback=","),
                      names=self.configs.get("common", "HEADERS").split(","),
                      engine='python', usecols=[0, 2, 3])
        self.ui_data_frame = pd.read_csv(
            self.get_file_path(self.configs.get(self._dataset_name, 'file_path'),
                               self.configs.get(self._dataset_name, 'file_name')),
            **params)
        self.ui_data_frame.fillna(value={"rating": -1}, inplace=True)

    def filter_df(self, filters: Dict[str, Any]) -> None:
        """
        filters: a dictionary whose keys are the name of the column on which
        the filtering is being applied;
        In case of age, the value must be a tuple, like (min_age, max_age)
        """
        headers = self.ui_data_frame.columns
        common_elements = np.intersect1d(list(filters.keys()), headers)
        if not common_elements:
            return
        df_filters = [(self.ui_data_frame[key].map(lambda x: str(x).lower()) ==
                       str(filters[key]).lower())
                      if key not in ['rating', 'age']
                      else (self.ui_data_frame[key].astype(int)
                            .between(int(filters[key][0]), int(filters[key][1]),
                                     inclusive = "both"))
                      for key in common_elements]
        self.ui_data_frame = self.ui_data_frame[ft.reduce(lambda x, y: (x) & (y), df_filters)]

    def df_add_column(self, add_columns: List[str]) -> None:
        """
        this function gets the list of columns to be added and add them to the dataframe
        """
        headers = ["user", "gender", "age", "country"]
        common_elements = np.intersect1d(add_columns, headers)
        if not common_elements:
            return
        # in profile.tsv there is more user-related info that is why we use this file 
        file_name = "usersha1-profile.tsv"
        fields_map = {"gender": 1, "age": 2, "country": 3}
        use_cols = [0]
        use_cols.extend([fields_map[col_name] for col_name in add_columns])
        headers = [headers[i] for i in sorted(use_cols)]
        params = dict(delimiter=self.configs.get("common", "DELIMITER", fallback=","),
                      names=headers, engine='python', usecols=use_cols)
        data_frame = pd.read_csv(
            self.get_file_path(self.configs.get(self._dataset_name, 'file_path'),
                               file_name), **params)
        self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame, on=["user"], how="left")
        values = {"gender": "", "age": -1, "Country": ""}
        self.ui_data_frame.fillna(value=values, inplace=True)


class Dataloader1B(DataloaderBase):
    """
    In order to load 1B
    """

    # the list of headers of files LFM-1b_{f_name}.txt where f_name is the key of this dictionary
    # http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf
    options = {"albums": ["album-id", "album-name", "artist-id"],
               "artists": ["artist-id", "artist-name"],
               "tracks": ["track-id", "track-name", "artist-id"],
               "LEs": ["user-id", "artist-id", "album-id", "track-id", "timestamp"],
               "users": ["user-id", "country", "age", "gender", "playcount",
                         "registered_timestamp"],
               "users_additional": ["user-id",
                                    # novelty score, percentage of new artists listened to,
                                    # averaged overtime windows of 1 month
                                    "novelty_artist_avg_month",
                                    "novelty_artist_avg_6months",
                                    "novelty_artist_avg_year",
                                    # mainstreaminess score,
                                    # overlap between the user’s listening history
                                    "mainstreaminess_avg_month",
                                    "mainstreaminess_avg_6months",
                                    "mainstreaminess_avg_year",
                                    "mainstreaminess_global",
                                    # total number of the user’s listening events
                                    # (playcounts) included in the dataset
                                    "cnt_listeningevents",
                                    "cnt_distinct_tracks",
                                    "cnt_distinct_artists",
                                    "cnt_listeningevents_per_week",
                                    # fraction of listening events for each weekday
                                    # (starting on Monday) among all weekly plays,
                                    # averaged over the user’s entire listening history
                                    "relative_le_per_weekday1",
                                    "relative_le_per_weekday2",
                                    "relative_le_per_weekday3",
                                    "relative_le_per_weekday4",
                                    "relative_le_per_weekday5",
                                    "relative_le_per_weekday6",
                                    "relative_le_per_weekday7",
                                    "relative_le_per_hour0",
                                    "relative_le_per_hour1",
                                    "relative_le_per_hour2",
                                    "relative_le_per_hour3",
                                    "relative_le_per_hour4",
                                    "relative_le_per_hour5",
                                    "relative_le_per_hour6",
                                    "relative_le_per_hour7",
                                    "relative_le_per_hour8",
                                    "relative_le_per_hour9",
                                    "relative_le_per_hour10",
                                    "relative_le_per_hour11",
                                    "relative_le_per_hour12",
                                    "relative_le_per_hour13",
                                    "relative_le_per_hour14",
                                    "relative_le_per_hour15",
                                    "relative_le_per_hour16",
                                    "relative_le_per_hour17",
                                    "relative_le_per_hour18",
                                    "relative_le_per_hour19",
                                    "relative_le_per_hour20",
                                    "relative_le_per_hour21",
                                    "relative_le_per_hour22",
                                    "relative_le_per_hour23"],
}

    def __init__(self, config_path: str = "") -> None:
        super().__init__(config_path)
        self._dataset_name = "LFM-1B"
        # we need to union the sets in order to get rid of duplicates 
        self._filter_options = ft.reduce(lambda x, y: set(x) | set(y), self.options.values())

    def load_data(self) -> None:
        """
        This function loads the mat_file
        in order to make user_artist_rating dataframe
        """
        mat_file = self.get_file_path(self.configs.get(self._dataset_name, 'file_path'),
                                      self.configs.get(self._dataset_name, 'file_name'))
        # Read the user-artist-matrix and corresponding artist and user indices from Matlab file
        # source: http://www.cp.jku.at/datasets/LFM-1b/LFM-1b_stats.py
        mat_data = h5py.File(mat_file, 'r')
        user_ids = [item[0] for item in np.array(mat_data.get('idx_users')).astype(np.int64)]
        artist_ids = np.array(mat_data.get('idx_artists')).astype(np.int64)
        ua_matrix = sparse.csr_matrix((mat_data['/LEs/']["data"],
                                       mat_data['/LEs/']["ir"],
                                       mat_data['/LEs/']["jc"])).transpose()
        for i, user_i in enumerate(user_ids):
            # pc_i represent the rows(users), idx_nz is the non-zero value (ratings)
            pc_i = ua_matrix.getrow(i).toarray()[0]
            idx_nz = np.nonzero(pc_i)
            ratings_i = pc_i[idx_nz]     # get playcount vector for user i
            artist_ids_i = [item[0] for item in artist_ids[idx_nz]]
            data_frame = list(zip([user_i] * len(artist_ids_i), artist_ids_i, ratings_i))
            data_frame = pd.DataFrame(data_frame, columns=['user', 'item', 'rating'])
            # if this is the first time 
            if self.ui_data_frame.empty:
                self.ui_data_frame = data_frame.copy()
            else:
                self.ui_data_frame = pd.concat([self.ui_data_frame, data_frame], ignore_index=True)

    def df_add_column(self, add_columns: List[str]) -> None:
        """
        this function adds list of columns to our data frame
        """
        # our dataframe's headers are user-item-rating
        # renaming columns to be able to join the data frames 
        self.ui_data_frame.rename(columns = {'user': 'user-id', 'item': 'artist-id'},
                                  inplace = True)
        # if album-name should be added we need LES and albums tables
        # to merge on user-id and artist-id (in general item related columns) 
        # and later be added to the dataframe 
        if "album-name" in add_columns:
            df_base = self.read_file("LEs", ["user-id", "artist-id", "album-id"])
            df_album = self.read_file("albums", self.options["albums"])
            data_frame = pd.merge(df_base, df_album, on=["artist-id", "album-id"], how="left")
            data_frame = data_frame[["user-id", "artist-id", "album-name"]]
            self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame,
                                          on=["user-id", "artist-id"],
                                          how="left")
            self.ui_data_frame.fillna(value={"album-name": ""}, inplace=True)
        if "track-name" in add_columns:
            df_base = self.read_file("LEs", ["user-id", "artist-id", "track-id"])
            df_track = self.read_file("tracks", self.options["tracks"])
            data_frame = pd.merge(df_base, df_track, on=["artist-id", "track-id"], how="left")
            data_frame = data_frame[["user-id", "artist-id", "track-name"]]
            self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame,
                                          on=["user-id", "artist-id"],
                                          how="left")
            self.ui_data_frame.fillna(value={"track-name": ""}, inplace=True)
        if "artist-name" in add_columns:
            df_base = self.read_file("LEs", ["user-id", "artist-id"])
            df_artist = self.read_file("artists", self.options["artists"])
            data_frame = pd.merge(df_base, df_artist, on=["artist-id"], how="left")
            data_frame = data_frame[["user-id", "artist-id", "artist-name"]]
            self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame,
                                          on=["user-id", "artist-id"],
                                          how="left")
            self.ui_data_frame.fillna(value={"artist-name": ""}, inplace=True)
        common_elements = list(np.intersect1d(add_columns, self.options['LEs']))
        if common_elements:
            # we need user-id for join operation
            if not "user-id" in common_elements:
                common_elements.insert(0, "user-id")
            data_frame = self.read_file("LEs", common_elements)
            self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame, on=["user-id"],
                                          how="left")
        common_elements = list(np.intersect1d(add_columns, self.options['users']))
        if common_elements:
            if "user-id" not in common_elements:
                common_elements.insert(0, "user-id")
            data_frame = self.read_file("users", common_elements)
            self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame, on=["user-id"],
                                          how="left")
            self.ui_data_frame.fillna(value={"country": "", "age": -1, "gender": "",
                                             "playcount": -1,
                                             "registered_timestamp": "01-01-1900"},
                                      inplace=True)
        common_elements = list(np.intersect1d(add_columns, self.options['users_additional']))
        if common_elements:
            if "user-id" not in common_elements:
                # user-id needs to be there to perform the join. if it is not among add_columns, I need to 
                # add it artificially and then omit the column because in the end I don't need user-id column
                common_elements.insert(0, "user-id")
            data_frame = self.read_file("users_additional", common_elements)
            self.ui_data_frame = pd.merge(self.ui_data_frame, data_frame, on=["user-id"],
                                          how="left")
            _ = common_elements.pop(0) #_wild card: we don't need the user-id anymore
            # we make a dictionary to mark the na_values as -1 to feed it to our data frame
            na_values = dict(zip(common_elements, [-1.0] * len(common_elements)))
            self.ui_data_frame.fillna(value=na_values, inplace=True)

        self.ui_data_frame.rename(columns = {'user-id': 'user', 'artist-id': 'item'},
                                  inplace = True)

    def read_file(self, f_name: str, r_columns: List[str]) -> pd.DataFrame:
        """
        this function takes the file name and list of columns to be read and returns the data frame
        example: LFM-1b_users.txt
        """
        file_name = f"LFM-1b_{f_name}.txt"
        params = dict(delimiter=self.configs.get("common", "DELIMITER", fallback=","),
                      header=0, engine='python', usecols=r_columns)
        data_frame = pd.read_csv(
            self.get_file_path(self.configs.get(self._dataset_name, 'file_path'),
                               file_name), **params)
        return data_frame

    def filter_df(self, filters: Dict[str, Any]) -> None:
        """
        this function takes a dictionary of string of any type(could be int, tuple,..) 
        and returns the updated data frame
        """
        headers = self.ui_data_frame.columns
        common_elements = list(np.intersect1d(list(filters.keys()), headers))
        if not common_elements:
            return
        range_features = ['rating', 'age', 'playcount', 'timestamp', 'registered_timestamp']
        range_features.extend(self.options['users_additional'][1:])
        df_filters = [(self.ui_data_frame[key].map(lambda x: str(x).lower()) ==
                       str(filters[key]).lower())
                      if key not in range_features
                      else (self.ui_data_frame[key].astype(int)
                            .between(int(filters[key][0]), int(filters[key][1]),
                                     inclusive = "both"))
                      for key in common_elements]
        self.ui_data_frame = self.ui_data_frame[ft.reduce(lambda x, y: (x) & (y), df_filters)]


def get_dataloader(dataset_name: str, config_path: str = "") -> DataloaderBase:
    """
    this function takes the name of the dataset and returns its dataloader.
    the config_path is optional
    """
    if dataset_name.upper() == "LFM-360K":
        return Dataloader360K(config_path)
    if dataset_name.upper() == "LFM-1B":
        return Dataloader1B(config_path)
    return None


def explore_datasets(config_file_path: str, data_dir: str) -> None:
    """
    this function takes the path of the config file and the directory contains the datasets
    and writes the configuration file
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
            # making the file
            with open(dataset_path, 'a', encoding="utf-8") as dfile:
                dfile.write('')
        # now for each section make the default value
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
            with open(config_file_path, 'a', encoding="utf-8") as configfile:
                config.write(configfile)
