""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)


For each selected dataset in an experiment this function is called once
    
    dataset:        pointer to the original dataset class
    train_path:     path where the used train set is stored
    test_path:      path where the used test set is stored
    model_dirs:     a list of all dirs for each calculated model, each dir contains a single file with the rating results
    eval_config:    the total config for evaluation: metrics and filters
    callback:       mostly consistency with other pipelines (maybe we should switch to a generic logger/callback class?)
    **kwargs:       whether they are for recommender experiments contains top_K in kwargs[num_items]
    
Could store the performance results in their respective model_dirs
"""
from fairreckitlib.metrics.lan import Test
from fairreckitlib.metrics.pipeline2 import EvaluationPipeline


def run_evaluation_pipelines(dataset, train_path, test_path, model_dirs, eval_config, callback, **kwargs):
    print('model_dirs:')
    print(model_dirs)

    for model_dir in model_dirs:
        import os
        print('model_dir:')
        print(model_dir)
        dir_name = os.path.dirname(model_dir)
        from fairreckitlib.metrics.metrics2 import RecType

        # Create a test instance TODO refactor
        test = Test(name=dir_name, train_path=train_path, test_path=test_path,
                    recs_path=model_dir+'/ratings.tsv', rec_type=RecType.Recommendation)

        pipeline = EvaluationPipeline(test, '', eval_config['metrics'], kwargs['num_items'], eval_config['filters'])
        pipeline.run()

    pass
