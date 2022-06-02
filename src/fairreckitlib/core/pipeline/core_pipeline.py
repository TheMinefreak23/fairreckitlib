"""This module contains the base core pipeline class.

Classes:

    CorePipeline:

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

import pandas as pd

from ..events.event_dispatcher import EventDispatcher
from ..events.event_error import ON_RAISE_ERROR, ErrorEventArgs
from ..io.event_io import ON_CREATE_FILE, DataframeEventArgs, FileEventArgs


class CorePipeline:
    """Base class for FairRecKit pipelines.

    This class exposes some reusable functionality that can be used in derived
    pipelines to read and/or write dataframes.
    """

    def __init__(self, event_dispatcher: EventDispatcher):
        """Construct the CorePipeline.

        Args:
            event_dispatcher: used to dispatch events when running the core pipeline.
        """
        self.event_dispatcher = event_dispatcher

    def read_dataframe(
            self,
            dataframe_path: str,
            dataframe_name: str,
            event_id_on_begin: str,
            event_id_on_end: str,
            *,
            names=None) -> pd.DataFrame:
        """Read a dataframe from the disk.

        This function dispatches an error event when the FileNotFoundError is raised,
        and thereafter the error is raised once more.

        Args:
            dataframe_path: path to the dataframe file.
            dataframe_name: name of the dataframe to use for event dispatching.
            event_id_on_begin: the event_id to dispatch when loading starts.
            event_id_on_end: the event_id to dispatch when loading is finished.
            names: the column names of the dataframe or None to infer them from the header.

        Returns:
            the loaded dataframe.
        """
        self.event_dispatcher.dispatch(DataframeEventArgs(
            event_id_on_begin,
            dataframe_path,
            dataframe_name
        ))

        start = time.time()

        try:
            dataframe = pd.read_csv(
                dataframe_path,
                sep='\t',
                header='infer' if names is None else None,
                names=names
            )
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_RAISE_ERROR,
                'FileNotFoundError: raised while trying to load the ' +
                dataframe_name + ' from ' + dataframe_path
            ))
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(DataframeEventArgs(
            event_id_on_end,
            dataframe_path,
            dataframe_name
        ), elapsed_time=end - start)

        return dataframe

    def write_dataframe(
            self,
            dataframe_path: str,
            dataframe: pd.DataFrame,
            header: bool) -> None:
        """Write a dataframe to the disk.

        This function is intended to write (append) a dataframe in chunks,
        including the header of the dataframe.
        It is assumed that when the header is True that the dataframe file
        has been created, which in turn will dispatch the IO event.

        Args:
            dataframe_path: path to the dataframe file.
            dataframe: the dataframe to append to the file.
            header: whether to include the header.
        """
        dataframe.to_csv(
            dataframe_path,
            mode='a',
            sep='\t',
            header=header,
            index=False
        )
        # header is the first line meaning the file has just been created
        if header:
            self.event_dispatcher.dispatch(FileEventArgs(ON_CREATE_FILE, dataframe_path))
