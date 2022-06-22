"""This package co

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_dir
from src.fairreckitlib.core.io.io_delete import delete_dir

from ...core.config.config_factories import GroupFactory
from ...core.pipeline.core_pipeline import CorePipeline

from .filter_config import DataSubsetConfig


CorePipeline(EventDispatcher())

def filter_from_filterpasses(CorePipeline: CorePipeline,
                             output_dir: str,
                             dataframe: pd.DataFrame,
                             subset: DataSubsetConfig,
                             filter_factory: GroupFactory):
    dir_path = create_dir(output_dir + 'filter_passes_temp', EventDispatcher())  # waht eventdispatche, where directory??? add path? add file?
    og_df_path = dir_path + 'og_df'
    filterpass_df_path = dir_path + 'filterpass_df'
    CorePipeline.write_dataframe(og_df_path, dataframe, True)
    filter_dataset_factory = filter_factory.get_factory(subset.dataset).getfactory(subset.matrix)
    for i, filter_pass_config in enumerate(subset.filter_passes): # enumerate dict:??
        # callable?  df
        dataframe = CorePipeline.read_dataframe(og_df_path, 'input_dataframe', 'start?', 'end?')  #???
        for filter in filter_pass_config:
            filterobj = filter_dataset_factory.create(filter.name, filter.params)
            dataframe = filterobj.run(dataframe)
        if i == 0:  # Whether to include header
            CorePipeline.write_dataframe(filterpass_df_path, dataframe, True)
        else:
            CorePipeline.write_dataframe(filterpass_df_path, dataframe, False)

    # remove dir
    dataframe = CorePipeline.read_dataframe(filterpass_df_path, 'filtered_dataframe', EventDispatcher())
    delete_dir(dir_path, EventDispatcher())
    return dataframe

