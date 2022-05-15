"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...core.config_constants import KEY_NAME
from ..filter.filter_constants import KEY_DATA_FILTERS
from ..ratings.convert_constants import KEY_RATING_CONVERTER
from ..ratings.convert_config import ConvertConfig
from ..split.split_constants import KEY_SPLITTING
from ..split.split_config import SplitConfig


@dataclass
class DatasetConfig:
    """Dataset Configuration."""

    name: str
    prefilters: []
    converter: Optional[ConvertConfig]
    splitting: SplitConfig

    def to_yml_format(self) -> Dict[str, Any]:
        """Format dataset configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset configuration.
        """
        yml_format = {
            KEY_NAME: self.name,
            KEY_SPLITTING: self.splitting.to_yml_format()
        }

        # only include prefilters if it has entries
        if len(self.prefilters) > 0:
            # TODO convert filters to yml format
            yml_format[KEY_DATA_FILTERS] = []

        # only include rating modifier if it is present
        if self.converter:
            yml_format[KEY_RATING_CONVERTER] = self.converter.to_yml_format()

        return yml_format
