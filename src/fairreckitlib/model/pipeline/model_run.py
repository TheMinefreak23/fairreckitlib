"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def run_model_pipelines(output_dir, data_transition, model_factory,
                        models_config, event_dispatcher, is_running, **kwargs):
    """Runs several ModelPipeline's for the specified model configurations.

    Args:
        output_dir(str): the path of the directory to store the output.
        data_transition(DataTransition): data input.
        model_factory(GroupFactory): the model factory with available algorithms.
        models_config(dict): containing list of ModelConfig's keyed by API name.
        event_dispatcher(EventDispatcher): used to dispatch model/IO events
            when running the model pipelines.
        is_running(func -> bool): function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Keyword Args:
        num_threads(int): the max number of threads a model can use.
        num_items(int): the number of item recommendations to produce, only
            needed when running recommender pipelines.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        model_dirs(array like): list of directories where the computed model
            ratings are stored.
    """
    model_dirs = []

    for api_name, models in models_config.items():
        if not is_running():
            return None

        api_factory = model_factory.get_factory(api_name)
        if api_factory is None:
            # TODO log this
            continue

        try:
            model_pipeline = api_factory.func_create_pipeline(api_factory, event_dispatcher)
            dirs = model_pipeline.run(
                output_dir,
                data_transition,
                models,
                is_running,
                **kwargs
            )
        except MemoryError:
            # TODO log this
            continue

        model_dirs += dirs

    return model_dirs
