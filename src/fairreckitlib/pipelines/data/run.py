""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .pipeline import DataPipeline


def run_data_pipeline(output_dir, data_registry, split_factory,
                      datasets_config, event_dispatcher):
    """Runs the Data Pipeline for multiple dataset configurations.

    Args:
        output_dir(str): the path of the directory to store the output.
        data_registry(DataRegistry): the registry of available datasets.
        split_factory(SplitFactory): factory of available splitters.
        datasets_config(array like): list of DatasetConfig objects.
        event_dispatcher(EventDispatcher): used to dispatch data/IO events
            when running the pipeline.

    Returns:
        data_result(array like): list of DataTransition's.
    """
    data_result = []

    data_pipeline = DataPipeline(split_factory, event_dispatcher)
    for _, data_config in enumerate(datasets_config):
        dataset = data_registry.get_set(data_config.name)

        data_result.append(data_pipeline.run(
            output_dir,
            dataset,
            data_config
        ))

    return data_result
