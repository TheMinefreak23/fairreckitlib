"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .pipeline import ModelPipeline


class RecommenderPipeline(ModelPipeline):
    """Recommender Pipeline that computes item recommendations.

    The topK item recommendations will be computed for each user that is present in the test set.
    """

    def test_model_ratings(self, model, output_path, batch_size, **kwargs):
        test_users = self.test_set.user.unique()
        start_index = 0
        while start_index < len(test_users):
            user_batch = test_users[start_index : start_index + batch_size]
            recommendations = model.recommend_batch(user_batch, num_items=kwargs['num_items'])
            recommendations.to_csv(output_path, mode='a', sep='\t', header=False, index=False)
            start_index += batch_size
