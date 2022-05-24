"""This modules contains the class to process the LastFM-360K dataset.

Classes:

    DatasetProcessorLFM360K: data processor implementation for the LFM-360K dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..dataset_config import create_dataset_table_config, DatasetMatrixConfig, DatasetTableConfig
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_lfm import DatasetProcessorLFM


class DatasetProcessorLFM360K(DatasetProcessorLFM):
    """DatasetProcessor for the LastFM-360K dataset.

    The dataset can be downloaded from the website below.
    https://www.upf.edu/web/mtg/lastfm360k

    The enriched artist gender information can be retrieved from:
    https://zenodo.org/record/3748787#.YowEBqhByUk

    The processor handles the following files:

    usersha1-artmbid-artname-plays.tsv (required)
    usersha1-profile.tsv (optional)
    lfm-360-gender.json (optional)
    """

    def __init__(self, dataset_dir: str, dataset_name: str):
        """Construct the DatasetProcessorLFM360K.

        Args:
            dataset_name: path of the dataset directory.
            dataset_name: name of the dataset (processor).
        """
        DatasetProcessorLFM.__init__(self, dataset_dir, dataset_name)
        # buffer for the user sha and artist name lists
        self.user_list = None
        self.artist_list = None
        # buffer for the artist name/musicbrainzID dataframe
        self.artist_mb_id = None

    def create_listening_events_config(self) -> Optional[DatasetTableConfig]:
        """Create the listening event table configuration.

        No listening events are available for this dataset.

        Returns:
            None.
        """
        return None

    def create_user_table_config(self) -> DatasetTableConfig:
        """Create the user table configuration.

        The base user configuration that contains the generated user ids
        and corresponding user sha.

        Returns:
            the configuration of the user table.
        """
        return create_dataset_table_config(
            TABLE_FILE_PREFIX + self.dataset_name + '_users.tsv.bz2',
            ['user_id'],
            ['user_sha'],
            compression='bz2',
            num_records=len(self.user_list)
        )

    def get_matrix_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetMatrixConfig]]]]:
        """Get matrix configuration processors.

        Returns:
            a list containing the user-artist-count matrix processor.
        """
        return [('user-artist-count', self.process_user_artist_matrix)]

    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Returns:
            a list containing the artist and user table processors.
        """
        return DatasetProcessorLFM.get_table_configs(self) + [('artist',self.process_artist_table)]

    def load_artist_gender_json(self) -> Optional[pd.DataFrame]:
        """Load the artist gender json file.

        Returns:
            the loaded artist musicbrainzID/gender table or None on failure.
        """
        try:
            gender_table = pd.read_json(
                os.path.join(self.dataset_dir, 'lfm-360-gender.json'),
                orient='index'
            )
            gender_table.reset_index(inplace=True)
            gender_table.rename(columns={'index': 'artist_mbID', 0: 'artist_gender'}, inplace=True)
            return gender_table
        except FileNotFoundError:
            return None

    def load_user_table(self) -> Optional[pd.DataFrame]:
        """Load the original user table.

        Changes the contents of the age and gender columns to be more user-friendly,
        and the contents of the country column to ISO 3166 Alpha-2 country codes.

        Returns:
            the loaded user table on None on failure.
        """
        user_table_columns = [
            'user_sha',
            'user_gender',
            'user_age',
            'user_country',
            'user_signup'
        ]

        try:
            # load original user table
            user_table = pd.read_table(
                os.path.join(self.dataset_dir, 'usersha1-profile.tsv'),
                names=user_table_columns,
                sep='\t'
            )
        except FileNotFoundError:
            return None

        # mask user age not between 1-100 as NaN
        user_table['user_age'].mask(user_table['user_age'].lt(1), inplace=True)
        user_table['user_age'].mask(user_table['user_age'].gt(100), inplace=True)

        # convert gender to more user-friendly names
        user_table['user_gender'].replace({'m': 'Male', 'f': 'Female'}, inplace=True)

        # convert country to ISO 3166 Alpha-2 country code
        user_table['user_country'].replace({
            'Afghanistan': 'AF', 'Albania': 'AL', 'Algeria': 'DZ', 'American Samoa': 'AS',
            'Andorra': 'AD', 'Angola': 'AO', 'Anguilla': 'AI', 'Antarctica': 'AQ',
            'Antigua and Barbuda': 'AG', 'Argentina': 'AR', 'Armenia': 'AM', 'Aruba': 'AW',
            'Australia': 'AU', 'Austria': 'AT', 'Azerbaijan': 'AZ',
            'Bahamas': 'BS', 'Bahrain': 'BH', 'Bangladesh': 'BD', 'Barbados': 'BB',
            'Belarus': 'BY', 'Belgium': 'BE', 'Belize': 'BZ', 'Benin': 'BJ', 'Bermuda': 'BM',
            'Bhutan': 'BT', 'Bolivia': 'BO', 'Bosnia and Herzegovina': 'BA', 'Botswana': 'BW',
            'Bouvet Island': 'BV', 'Brazil': 'BR', 'British Indian Ocean Territory': 'IO',
            'Brunei Darussalam': 'BN', 'Bulgaria': 'BG', 'Burkina Faso': 'BF', 'Burundi': 'BI',
            'Cambodia': 'KH', 'Cameroon': 'CM', 'Canada': 'CA', 'Cape Verde': 'CV',
            'Cayman Islands': 'KY', 'Central African Republic': 'CF', 'Chad': 'TD', 'Chile': 'CL',
            'China': 'CN', 'Christmas Island': 'CX', 'Cocos (Keeling) Islands': 'CC',
            'Colombia': 'CO', 'Comoros': 'KM', 'Congo': 'CG', 'Czech Republic': 'CZ',
            'Congo, the Democratic Republic of the': 'CD', 'Cook Islands': 'CK', 'Cyprus': 'CY',
            'Costa Rica': 'CR', 'Cote D\'Ivoire': 'CI', 'Croatia': 'HR', 'Cuba': 'CU',
            'Denmark': 'DK', 'Djibouti': 'DJ', 'Dominica': 'DM', 'Dominican Republic': 'DO',
            'Ecuador': 'EC', 'Egypt': 'EG', 'El Salvador': 'SV', 'Equatorial Guinea': 'GQ',
            'Eritrea': 'ER', 'Estonia': 'EE', 'Ethiopia': 'ET',
            'Falkland Islands (Malvinas)': 'FK', 'Faroe Islands': 'FO', 'Fiji': 'FJ',
            'Finland': 'FI', 'France': 'FR', 'French Guiana': 'GF', 'French Polynesia': 'PF',
            'French Southern Territories': 'TF',
            'Gabon': 'GA', 'Gambia': 'GM', 'Georgia': 'GE', 'Germany': 'DE', 'Ghana': 'GH',
            'Grenada': 'GD', 'Gibraltar': 'GI', 'Greece': 'GR', 'Greenland': 'GL',
            'Guadeloupe': 'GP', 'Guam': 'GU', 'Guatemala': 'GT', 'Guinea-Bissau': 'GW',
            'Guyana': 'GY',
            'Haiti': 'HT', 'Heard Island and Mcdonald Islands': 'HM', 'Hungary': 'HU',
            'Holy See (Vatican City State)': 'VA', 'Honduras': 'HN', 'Hong Kong': 'HK',
            'Iceland': 'IS', 'India': 'IN', 'Indonesia': 'ID', 'Iran, Islamic Republic of': 'IR',
            'Iraq': 'IQ', 'Ireland': 'IE', 'Israel': 'IL', 'Italy': 'IT',
            'Jamaica': 'JM', 'Japan': 'JP', 'Jordan': 'JO',
            'Kazakhstan': 'KZ', 'Kenya': 'KE', 'Kiribati': 'KI', 'Kyrgyzstan': 'KG',
            'Korea, Democratic People\'s Republic of':'KP','Korea, Republic of':'KR','Kuwait':'KW',
            'Lao People\'s Democratic Republic': 'LA','Latvia': 'LV','Lebanon': 'LB',
            'Lesotho': 'LS', 'Liberia': 'LR', 'Libyan Arab Jamahiriya': 'LY',
            'Liechtenstein': 'LI', 'Lithuania': 'LT', 'Luxembourg': 'LU',
            'Macao': 'MO', 'Macedonia': 'MK', 'Madagascar': 'MG', 'Mali': 'ML', 'Malta': 'MT',
            'Malaysia': 'MY', 'Malawi': 'MW', 'Maldives': 'MV', 'Mauritania': 'MR',
            'Mauritius': 'MU', 'Marshall Islands': 'MH', 'Martinique': 'MQ', 'Mayotte': 'YT',
            'Mexico': 'MX', 'Micronesia, Federated States of': 'FM','Moldova': 'MD','Monaco': 'MC',
            'Mongolia': 'MN', 'Montenegro': 'ME', 'Montserrat':  'MS', 'Morocco': 'MA',
            'Mozambique': 'MZ', 'Myanmar': 'MM',
            'Namibia': 'NA', 'Nauru': 'NR', 'Nepal': 'NP',
            'Netherlands': 'NL', 'Netherlands Antilles': 'NL', 'New Caledonia': 'NC',
            'New Zealand': 'NZ', 'Nicaragua': 'NI', 'Niger': 'NE', 'Nigeria': 'NG', 'Niue': 'NU',
            'Norfolk Island': 'NF', 'Northern Mariana Islands': 'MP', 'Norway': 'NO',
            'Oman': 'OM',
            'Pakistan': 'PK', 'Palau': 'PW','Palestinian Territory, Occupied': 'PS','Panama': 'PA',
            'Papua New Guinea': 'PG', 'Paraguay': 'PY', 'Peru': 'PE', 'Philippines': 'PH',
            'Pitcairn': 'PN', 'Poland': 'PL', 'Portugal': 'PT', 'Puerto Rico': 'PR',
            'Qatar': 'QA',
            'Reunion': 'RE', 'Romania': 'RO', 'Russian Federation': 'RU', 'Rwanda': 'RW',
            'Saint Helena': 'SH', 'Saint Kitts and Nevis': 'KN', 'Saint Lucia': 'LC',
            'Saint Pierre and Miquelon': 'PM', 'Saint Vincent and the Grenadines': 'VC',
            'Samoa': 'WS', 'San Marino': 'SM', 'Sao Tome and Principe': 'ST', 'Saudi Arabia': 'SA',
            'Senegal': 'SN', 'Serbia': 'RS', 'Seychelles': 'SC', 'Sierra Leone': 'SL',
            'Singapore': 'SG', 'Solomon Islands': 'SB', 'Somalia': 'SO', 'South Africa': 'ZA',
            'Slovakia': 'SK','Slovenia': 'SI','South Georgia and the South Sandwich Islands': 'GS',
            'Spain': 'ES', 'Sri Lanka': 'LK', 'Sudan': 'SD', 'Suriname': 'SR',
            'Svalbard and Jan Mayen': 'SJ', 'Syrian Arab Republic': 'SY', 'Swaziland': 'SZ',
            'Sweden': 'SE', 'Switzerland': 'CH',
            'Taiwan': 'TW', 'Tajikistan': 'TJ', 'Tanzania, United Republic of': 'TZ',
            'Thailand': 'TH', 'Timor-Leste': 'TL', 'Togo': 'TG', 'Tokelau': 'TK', 'Tonga': 'TO',
            'Trinidad and Tobago': 'TT', 'Tunisia': 'TN', 'Turkey': 'TR', 'Turkmenistan': 'TM',
            'Turks and Caicos Islands': 'TC', 'Tuvalu': 'TV',
            'Uganda': 'UG', 'Ukraine': 'UA', 'United Arab Emirates': 'AE',
            'United Kingdom':'GB','United States':'US','United States Minor Outlying Islands':'UM',
            'Uruguay': 'UY', 'Uzbekistan': 'UZ',
            'Vanuatu': 'VU', 'Venezuela': 'VE', 'Viet Nam': 'VN', 'Virgin Islands, British': 'VG',
            'Virgin Islands, U.s.': 'VI',
            'Yemen': 'YE',
            'Wallis and Futuna': 'WF', 'Western Sahara': 'EH',
            'Zambia': 'ZM', 'Zimbabwe': 'ZW'
        }, inplace=True)

        return user_table

    def process_artist_table(self) -> Optional[DatasetTableConfig]:
        """Process the artist table.

        Creates the artist table with the musicbrainzID and gender information when available.

        Returns:
            the artist table configuration or None on failure.
        """
        artist_key = ['artist_id']
        artist_columns = ['artist_name']

        # connect artist id to name
        artist_table = pd.DataFrame(
            list(enumerate(self.artist_list)),
            columns=artist_key + artist_columns
        )

        # merge the artist musicbrainzID on name
        artist_table = pd.merge(artist_table, self.artist_mb_id, how='left', on='artist_name')
        artist_table['artist_mbID'].fillna(-1, inplace=True)
        artist_columns += ['artist_mbID']

        artist_gender = self.load_artist_gender_json()
        if artist_gender is not None:
            # merge artists with gender and update columns
            artist_table = pd.merge(artist_table, artist_gender, how='left', on='artist_mbID')
            artist_columns += ['artist_gender']

        # create artist table configuration
        artist_table_config = create_dataset_table_config(
            TABLE_FILE_PREFIX + self.dataset_name + '_artists.tsv.bz2',
            artist_key,
            artist_columns,
            compression='bz2',
            num_records=len(self.artist_list)
        )

        # store the generated artist table
        artist_table_config.save_table(artist_table, self.dataset_dir)

        return artist_table_config

    def process_user_artist_matrix(self) -> Optional[DatasetMatrixConfig]:
        """Process the user-artist-count matrix.

        The user-item matrix is stored in a file that also contains a musicbrainzID.
        The users are hashes and the items are names, both are converted to integers
        to comply to the CSR compatible format. In addition, any rows that contain
        corrupt data are removed in the process.

        Returns:
            the matrix configuration or None on failure.
        """
        try:
            dataframe = pd.read_table(
                os.path.join(self.dataset_dir, 'usersha1-artmbid-artname-plays.tsv'),
                names=['user_sha', 'artist_mbID', 'artist_name', 'matrix_count']
            )
        except FileNotFoundError:
            return None

        # remove rows from a user that is not a hash
        dataframe = dataframe[dataframe['user_sha'] != 'sep 20, 2008']

        # map users/items to category and ratings to be floating-point
        dataframe['user_sha'] = dataframe['user_sha'].astype("category")
        dataframe['artist_name'] = dataframe['artist_name'].astype("category")
        dataframe['matrix_count'] = dataframe['matrix_count'].astype(float)

        # remove rows that contain items that failed to map to category
        dataframe = dataframe[dataframe['artist_name'].cat.codes >= 0]
        # remove rows that have unusable ratings
        dataframe = dataframe[dataframe['matrix_count'] > 0]

        dataframe.drop_duplicates(subset=['user_sha', 'artist_name'], inplace=True)
        print(dataframe)
        # extract user/item indirection arrays
        self.user_list = list(dataframe['user_sha'].cat.categories)
        self.artist_list = list(dataframe['artist_name'].cat.categories)

        # extract artist name/musicbrainzID combinations
        self.artist_mb_id = dataframe[['artist_name', 'artist_mbID']]
        # remove duplicates combinations
        self.artist_mb_id = self.artist_mb_id.drop_duplicates().dropna()
        # remove duplicates where the artist has more than one musicbrainzID
        self.artist_mb_id = self.artist_mb_id.drop_duplicates(subset='artist_name')

        # add the correct user/item integers
        dataframe['user_id'] = dataframe['user_sha'].cat.codes.copy()
        dataframe['artist_id'] = dataframe['artist_name'].cat.codes.copy()

        # create matrix by removing other columns
        user_artist_matrix = dataframe[['user_id', 'artist_id', 'matrix_count']]
        user_artist_matrix_table_config = create_dataset_table_config(
            TABLE_FILE_PREFIX + self.dataset_name + '_user-artist-count_matrix.tsv.bz2',
            ['user_id', 'artist_id'],
            ['matrix_count'],
            compression='bz2',
            foreign_keys=['user_id', 'artist_id']
        )

        # store the resulting matrix
        user_artist_matrix_table_config.save_table(user_artist_matrix, self.dataset_dir)

        return self.process_matrix(user_artist_matrix_table_config)

    def process_user_table(self) -> Optional[DatasetTableConfig]:
        """Process the user table.

        Extends the original user table with unique user ids.

        Returns:
            the user table configuration or None on failure.
        """
        user_table_config = self.create_user_table_config()
        # connect user id to sha
        user_sha_ids = pd.DataFrame(
            list(enumerate(self.user_list)),
            columns=['user_id', 'user_sha']
        )

        # load original user table and when available, add it to user id/sha
        user_table = self.load_user_table()
        if user_table is None:
            user_table = user_sha_ids
        else:
            for i in range(1, len(user_table.columns)):
                user_table_config.columns += [user_table.columns[i]]

            # join user table with user ids
            user_table = pd.merge(user_sha_ids, user_table, how='left', on='user_sha')

            # fill unknown user age with -1 and cast back to int
            user_table['user_age'].fillna(-1.0, inplace=True)
            user_table['user_age'] = user_table['user_age'].astype(int)

        # store the generated user table
        user_table_config.save_table(user_table, self.dataset_dir)

        return user_table_config
