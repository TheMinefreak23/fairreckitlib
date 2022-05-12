"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from .model_pipeline import ModelPipeline, write_computed_ratings


class RecommendationPipeline(ModelPipeline):
    """Recommendation Pipeline that computes item recommendations.

    The topK item recommendations will be computed for each user that is present in the test set.
    """

    def get_ratings_dataframe(self):
        return pd.concat([self.train_set, self.test_set])

    def test_model_ratings(self, model, output_path, batch_size, is_running, **kwargs):
        test_users = self.test_set.user.unique()
        start_index = 0
        while start_index < len(test_users):
            if not is_running():
                return

            user_batch = test_users[start_index : start_index + batch_size]
            recommendations = model.recommend_batch(user_batch, num_items=kwargs['num_items'])
            if not is_running():
                return

            write_computed_ratings(output_path, recommendations, start_index==0)
            start_index += batch_size
