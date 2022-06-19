"""This package contains the fairreckitlib test suite.

Modules:

    conftest: constants and fixtures that are shared among all files in the test suite.
    test_core_config_factories: test the functionality of all (base) types of config factories.
    test_core_config_params: test the configuration parameters functionality.
    test_core_config_yml_obj: test the yml and object configuration functionality.
    test_core_events: test the core event dispatching/listening functionality.
    test_core_io: test the core IO functionality.
    test_core_parsing_assert: test the assertion parsing functionality.
    test_core_parsing_obj_params: test the object (parameters) parsing functionality.
    test_core_pipeline: test the core pipeline functionality.
    test_core_threading: test the threading base/processor functionality.
    test_data_config: test the formatting/parsing of the data (matrix) configurations.
    test_data_converting: test the dataframe rating conversion functionality.
    test_data_converting_config: test the formatting/parsing of the dataset converting config.
    test_data_factory: test the data (modifier) factories.
    test_data_filter_basic: test the dataframe filter functionality for the basic filters.
    test_data_filter_samples: test the basic filters with dataset samples.
    test_data_pipeline: test the data pipeline functionality.
    test_data_registry: test the data registry.
    test_data_splitting: test the dataframe splitting functionality.
    test_data_splitting_config: test the formatting/parsing of the dataset splitting config.
    test_data_subset_config: test the formatting/parsing of the data subset configuration.
    test_dataset: test the dataset wrapper functionality of the dataset configuration.
    test_evaluation_config: test the formatting/parsing of the eval config of the experiment.
    test_evaluation_factory: test the evaluation (type/category) factories.
    test_evaluation_metrics: test the interface of metrics.
    test_evaluation_pipeline: test the evaluation pipeline functionality.
    test_experiment_config: test the formatting/parsing of the experiment configurations.
    test_experiment_pipeline: test the experiment pipeline functionality.
    test_experiment_thread: test the experiment threading functionality.
    test_model_algorithm_matrices: test the interface of the algorithm matrices.
    test_model_algorithms: test the interface of predictors/recommenders.
    test_model_config: test the formatting/parsing of the model config of the experiment.
    test_model_factory: test the model (type/API) factories.
    test_model_pipeline: test the model pipeline functionality.
    test_recommender_system: test the top level functionality of the recommender system.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
