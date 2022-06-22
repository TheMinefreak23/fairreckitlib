"""This module contains a function that performs filtering from filter passes.

Functions:
    filter_from_filter_passes: Apply filter to filter passes.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import random
import pandas as pd

from ...core.events.event_dispatcher import EventDispatcher
from ...core.io.io_create import create_dir
from ...core.io.io_delete import delete_dir

from ...core.config.config_factories import GroupFactory
from ...core.pipeline.core_pipeline import CorePipeline

from .filter_config import DataSubsetConfig


def filter_from_filter_passes(core_pipeline: CorePipeline,
                              output_dir: str,
                              dataframe: pd.DataFrame,
                              subset: DataSubsetConfig,
                              filter_factory: GroupFactory) -> pd.DataFrame:
    """Apply filter to filter passes inside DataSubsetConfig.

    For each filter pass, a filtered dataframe is returned. After which the dataframes
    are appended to each other and returned.

    Args:
        core_pipeline: Pipeline where this function is used. Required for IO actions.
        output_dir: Directory to write temp dataframes to.
        dataframe: Dataframe to be filtered.
        subset: Configuration file containing filter passes.
        filter_factory: Factory containing filters.

    Returns:
        An aggregation of filtered dataframes. 
    """
    # Create temp files and store base dataframe
    random_num_str = str(random.randint(0, 100000))  # To prevent concurrency issues
    dir_path = create_dir(os.path.join(output_dir, 'filter_passes_temp'), EventDispatcher())  # waht eventdispatche, where directory??? add path? add file.txt?
    og_df_path = os.path.join(dir_path, 'og_df' + random_num_str + '.tsv')
    core_pipeline.write_dataframe(og_df_path, dataframe, True)
    final_df = None
    # Apply filter passes
    filter_dataset_factory = filter_factory.get_factory(subset.dataset).get_factory(subset.matrix)
    for i, filter_pass_config in enumerate(subset.filter_passes):
        dataframe = core_pipeline.read_dataframe(og_df_path, 'input_dataframe', 'start?', 'end?')  #???
        for _filter in filter_pass_config.filters:
            filterobj = filter_dataset_factory.create(_filter.name, _filter.params)
            dataframe = filterobj.run(dataframe)
        if i == 0:  # Whether to include header
            final_df = pd.concat([final_df, dataframe], copy=False).drop_duplicates().reset_index(drop=True)
        else:
            core_pipeline = pd.concat([final_df, dataframe], copy=False).drop_duplicates().reset_index(drop=True)

    delete_dir(dir_path, EventDispatcher())
    return final_df
