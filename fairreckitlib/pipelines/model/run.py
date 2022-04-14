"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def run_model_pipelines(output_dir, data_transition, model_factory,
                        models_config, event_dispatcher, **kwargs):
    """Runs several ModelPipeline's for the specified model configurations.

    Args:
        output_dir(str): the path of the directory to store the output.
        data_transition(DataTransition): data input.
        model_factory(ModelFactory): the model factory with available algorithms.
        models_config(dict): containing model configurations keyed by API.
        event_dispatcher(EventDispatcher): used to dispatch model/IO events
            when running the model pipelines.

    Keyword Args:
        num_threads(int): the max number of threads a model can use.
        num_items(int): the number of item recommendations to produce, only
            needed when running recommender pipelines.

    Returns:
        model_dirs(array like): list of directories where the computed model
            ratings are stored.
    """
    model_dirs = []

    for api_name, models in models_config.items():
        model_pipeline = model_factory.create_pipeline(api_name, event_dispatcher)
        model_dirs += model_pipeline.run(
            output_dir,
            data_transition,
            models,
            **kwargs
        )

    return model_dirs
