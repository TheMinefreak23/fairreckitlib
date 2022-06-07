"""This module tests the core pipeline functionality.

Fixtures:

    fixture_df_event_dispatcher: event dispatcher that prints dataframe/IO events.

Functions:

    test_core_pipeline_read_error: test core pipeline dataframe read FileNotFoundError.
    test_core_pipeline_read_write: test core pipeline dataframe read and write functions.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd
import pytest

from src.fairreckitlib.core.events.event_args import EventArgs
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.event_io import \
    ON_CREATE_FILE, get_io_event_print_switch, print_load_df_event_args
from src.fairreckitlib.core.pipeline.core_pipeline import CorePipeline

ON_BEGIN_READ_DATAFRAME = 'DataFrame.on_begin_read'
ON_END_READ_DATAFRAME = 'DataFrame.on_end_read'

DATAFRAME_EVENTS = [ON_CREATE_FILE, ON_BEGIN_READ_DATAFRAME, ON_END_READ_DATAFRAME]


@pytest.fixture(scope='function', name='df_event_dispatcher')
def fixture_df_event_dispatcher() -> EventDispatcher:
    """Fix dataframe event dispatcher creation for core pipeline test functions."""
    def print_df_io_event(_, event_args: EventArgs, **kwargs) -> None:
        """Print the dataframe/IO event using print switches."""
        elapsed_time = kwargs.get('elapsed_time')
        print_switch = {
            ON_BEGIN_READ_DATAFRAME: print_load_df_event_args,
            ON_END_READ_DATAFRAME: lambda args: print_load_df_event_args(args, elapsed_time)
        }
        print_switch.update(get_io_event_print_switch())
        print_switch[event_args.event_id](event_args)

    event_dispatcher = EventDispatcher()
    for event_id in DATAFRAME_EVENTS:
        event_dispatcher.add_listener(event_id, None, (print_df_io_event, None))

    return event_dispatcher


def test_core_pipeline_read_error(io_tmp_dir: str, df_event_dispatcher: EventDispatcher) -> None:
    """Test the core pipeline dataframe read error for an unknown file."""
    pipeline = CorePipeline(df_event_dispatcher)

    with pytest.raises(FileNotFoundError):
        pipeline.read_dataframe(
            os.path.join(io_tmp_dir, 'unknown.tsv'),
            'dataframe',
            ON_BEGIN_READ_DATAFRAME,
            ON_END_READ_DATAFRAME
        )


@pytest.mark.parametrize('use_header', [True, False])
def test_core_pipeline_read_write(
        use_header: bool,
        io_tmp_dir: str,
        df_event_dispatcher: EventDispatcher) -> None:
    """Test the core pipeline dataframe read and write functions."""
    dataframe_name = 'dataframe'
    dataframe_path = os.path.join(io_tmp_dir, dataframe_name + '.tsv')

    column_range = range(0, 100)
    dataframe_chunk = pd.DataFrame()
    dataframe_chunk['ints'] = list(column_range)
    dataframe_chunk['floats'] = [float(i) for i in column_range]
    dataframe_chunk['strings'] = [str(i) + '_' for i in column_range]

    current_dataframe = dataframe_chunk
    pipeline = CorePipeline(df_event_dispatcher)

    for i in range(0, 10):
        is_file = os.path.isfile(dataframe_path)
        assert not is_file if i == 0 else is_file, \
            'expected dataframe file to be present after the first loop'

        # append chunk to dataframe file
        pipeline.write_dataframe(
            dataframe_path,
            dataframe_chunk,
            header=use_header and i == 0
        )

        # read back in the appended dataframe file
        read_dataframe = pipeline.read_dataframe(
            dataframe_path,
            dataframe_name,
            ON_BEGIN_READ_DATAFRAME,
            ON_END_READ_DATAFRAME,
            names=None if use_header else ['ints', 'floats', 'strings']
        )

        assert current_dataframe.eq(read_dataframe).all().all(), \
            'expected current dataframe to be the same as the read dataframe'

        # update current dataframe with the appended chunk
        current_dataframe = pd.concat([current_dataframe, dataframe_chunk], ignore_index=True)
