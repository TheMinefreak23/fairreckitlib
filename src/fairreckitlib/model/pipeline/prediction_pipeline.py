"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .model_pipeline import ModelPipeline, write_computed_ratings


class PredictionPipeline(ModelPipeline):
    """Prediction Pipeline that computes user/item rating predictions.

    The (user,item) prediction will be computed and for each pair that is present in the test set.
    """

    def get_ratings_dataframe(self):
        return self.test_set

    def test_model_ratings(self, model, output_path, batch_size, is_running, **kwargs):
        start_index = 0
        while start_index < len(self.test_set):
            if not is_running():
                return

            user_batch = self.test_set[start_index : start_index + batch_size]
            predictions = model.predict_batch(user_batch)
            if not is_running():
                return

            write_computed_ratings(output_path, predictions, start_index==0)
            start_index += batch_size
