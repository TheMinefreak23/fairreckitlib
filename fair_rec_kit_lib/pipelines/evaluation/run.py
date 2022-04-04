""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

"""
For each selected dataset in an experiment this function is called once
    
    dataset:        pointer to the original dataset class
    train_path:     path where the used train set is stored
    test_path:      path where the used test set is stored
    model_dirs:     a list of all dirs for each calculated model, each dir contains a single file with the rating results
    eval_config:    the total config for evaluation, still needs to be specified in what format it comes in
    callback:       mostly consistency with other pipelines (maybe we should switch to a generic logger/callback class?)
    **kwargs:       for recommender experiments contains top_K in kwargs[num_items]
    
Could store the performance results in their respective model_dirs
"""
def run_evaluation_pipelines(dataset, train_path, test_path, model_dirs, eval_config, callback, **kwargs):
    pass
