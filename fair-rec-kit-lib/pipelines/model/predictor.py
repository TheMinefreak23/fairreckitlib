""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .pipeline import ModelPipeline


class PredictorPipeline(ModelPipeline):

    def _run_model_test(self, model, callback, **kwargs):
        raise NotImplementedError()
