""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import yaml

from .pipeline import ModelPipeline


class EvaluationPipeline(ModelPipeline):

    def _run_model_test(self, model, callback, **kwargs):
        print('Evaluating...')
        recs = model.recommend(1, num_items=kwargs['num_items'])
        print(recs)
        print('')


class EvaluationPipelineElliot(EvaluationPipeline):

    def __init__(self, api_name, factory):
        EvaluationPipeline.__init__(self, api_name, factory)

    def run(self, train_set_path, test_set_path, models, callback, **kwargs):
        callback.on_begin_pipeline(self._api_name)

        raise NotImplementedError()

        callback.on_end_pipeline(self._api_name)
