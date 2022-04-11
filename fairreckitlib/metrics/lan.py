from fairreckitlib.metrics.filter import Filter
from fairreckitlib.metrics.metrics2 import Metric, RecType
from fairreckitlib.metrics.pipeline2 import EvaluationPipeline


class Test:
    def __init__(self, name, recs_path, test_path, train_path, rec_type):
        self.name = name
        self.recs_path = recs_path
        self.test_path = test_path
        self.train_path = train_path
        self.rec_type = rec_type


test = RecType.Recommendation

test1_item_knn = Test(
    name='MovieLens 1m ItemKNN'
    , recs_path='D:/uu/Softwareproject/data/movielens_1m/recs/ItemKNN_nn=50_sim=cosine_imp=standard_bin'
              '=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv '
    , test_path='D:/uu/Softwareproject/data/movielens_1m/split/0/test.tsv'
    , train_path='D:/uu/Softwareproject/data/movielens_1m/split/0/train.tsv'
    , rec_type=RecType.Recommendation
)

test1B_als = Test(
    name= 'LFM 1B ALS'
    , recs_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/Implicit_AlternatingLeastSquares_0/ratings.tsv'
    , train_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/train_set.tsv'
    , test_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/test_set.tsv'
    , rec_type=RecType.Recommendation
)

test360_als = Test(
    name= 'LFM 360K ALS'
    , recs_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/Implicit_AlternatingLeastSquares_0/ratings.tsv'
    , train_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/train_set.tsv'
    , test_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/test_set.tsv'
    , rec_type=RecType.Recommendation
)

test360_pop = Test(
    name='LFM 360K PopScore'
    ,recs_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/LensKit_PopScore_0/ratings.tsv'
    , train_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/train_set.tsv'
    , test_path='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/test_set.tsv'
    , rec_type=RecType.Recommendation
)

test360_profile = 'D:/Downloads/lastfm-dataset-360K/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv'

must_have_metrics = [Metric.ndcg, Metric.precision, Metric.recall, Metric.rmse, Metric.mae, Metric.mrr]
should_have_metrics = [Metric.item_coverage, Metric.dir, Metric.pairwise_fairness, Metric.gini,
                       Metric.intra_list_similarity, Metric.similarity_cos, Metric.similarity_euclid, Metric.novelty]
test_result = test360_als

# Mock
# metrics = [Metric.ndcg]
metrics = [Metric.precision, Metric.recall, Metric.mrr, Metric.rmse, Metric.item_coverage, Metric.user_coverage]
k = 10
gender_filter = {'type': Filter.Equals.value, 'name': 'gender', 'value': 'male'}
country_filter = {'type': Filter.Equals.value, 'name': 'country', 'value': 'Mexico'}
age_filter = {'type': Filter.Clamp.value, 'name': 'age', 'value': {'min': 15, 'max': 25}}

# Filters per which we run the evaluation
filters = [
    [gender_filter],
    [country_filter, age_filter]  # Multiple filter 'passes'
]

pipeline = EvaluationPipeline(test_result, test360_profile, metrics, k, filters)
pipeline.run()
