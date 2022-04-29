"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .pipeline import ModelPipeline


class PredictorPipeline(ModelPipeline):
    """Prediction Pipeline that computes user/item rating predictions.

    The (user,item) prediction will be computed and for each pair that is present in the test set.
    """

    def test_model_ratings(self, model, output_path, batch_size, is_running, **kwargs):
        start_index = 0
        while start_index < len(self.test_set):
            if not is_running():
                return

            user_batch = self.test_set[start_index : start_index + batch_size]
            predictions = model.predict_batch(user_batch)
            predictions.to_csv(output_path, mode='a', sep='\t', header=False, index=False)
            start_index += batch_size
