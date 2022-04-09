from datetime import datetime

import pandas as pd
import enum


class Test:
    def __init__(self, recsPath, testPath, trainPath):
        self.recsPath = recsPath
        self.testPath = testPath
        self.trainPath = trainPath

    recsPath = ''
    testPath = ''
    trainPath = ''


test1_pred = Test(
    recsPath='D:/uu/Softwareproject/data/movielens_1m/recs/ItemKNN_nn=50_sim=cosine_imp=standard_bin'
             '=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv '
    , testPath='D:/uu/Softwareproject/data/movielens_1m/split/0/test.tsv'
    , trainPath='D:/uu/Softwareproject/data/movielens_1m/split/0/train.tsv')

test2_rec = Test(
    recsPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/Implicit_AlternatingLeastSquares_0/ratings.tsv'
    , trainPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/train_set.tsv'
    , testPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/test_set.tsv'
)

test3_pred = Test(
    recsPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/LensKit_PopScore_0/ratings.tsv'
    , trainPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/train_set.tsv'
    , testPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/test_set.tsv'
)



def rexmex_test():
        # Rexmex
        recs = pd.read_csv(test_result.recsPath, header=None, sep='\t', names=["source_id", "target_id", "y_score"])
        test = pd.read_csv(test_result.testPath, header=None, sep='\t', names=["source_id", "target_id", "y_true"])
        # Merge on user ID
        scores = pd.merge(recs, test[['source_id', 'y_true']], on=['source_id'])
        print(scores.head())

        train = pd.read_csv(test_result.trainPath, header=None, sep='\t', names=["user", "item", "rating"])

        # rating: works for predictors
        from rexmex.metrics.rating import mean_absolute_error

        mae = mean_absolute_error(scores['y_true'], scores['y_score'])
        print('mae:', mae)

        from rexmex.metrics.coverage import item_coverage

        # user, item columns to list of tuples
        tuple_recs = [tuple(r) for r in recs[['source_id', 'target_id']].to_numpy()]
        print(tuple_recs[0:10])
        possible_users_items = (train['user'], train['item'])
        it_coverage = item_coverage(possible_users_items, tuple_recs)
        usr_coverage = item_coverage(possible_users_items, tuple_recs)
        print('item coverage:', it_coverage, 'user coverage:', usr_coverage)

        # ranking: works for recommenders
        from rexmex.metrics.ranking import average_precision_at_k

        k = 10
        # NOTE TODO score doesn't make sense here, we need rankings
        precision_value = average_precision_at_k(scores['source_id'], scores['y_true'], k)
        print('precision:', precision_value)

        # from rexmex.metrics.classification import pr_auc_score

        # pr_auc_value = pr_auc_score(scores["y_true"], scores["y_score"])
        # print("{:.3f}".format(pr_auc_value))

        # from rexmex.metricset import ClassificationMetricSet

        # metric_set = ClassificationMetricSet()
        # metric_set.print_metrics()

        # from rexmex.scorecard import ScoreCard

        # score_card = ScoreCard(metric_set)
        # report = score_card.generate_report(scores)
        # print(report)


def lenskit_test():
    # Lenskit
    recs = pd.read_csv(test_result.recsPath, header=None, sep='\t', names=['user','item','score'])

    recs['rank'] = recs.groupby('user')['score'].rank()
    recs['Algorithm']='APPROACHNAME'
    print(recs.head())

    from lenskit import topn

    test_data = pd.read_csv(test_result.testPath, header=None, sep='\t', names=['user','item','rating'])

    # Concatenate into single frame
    #test_data = pd.concat(test_data, ignore_index=True)

    k = 10
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg, k=k)
    results = rla.compute(recs, test_data)
    print(results.head())
    print(results.groupby('Algorithm').ndcg.mean())

evalMethod = EvalMethod.Rexmex
test_result = test2_rec

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print('Starting evaluation..')

#print('Rexmex:')
#rexmex_test()
print('Lenskit:')
lenskit_test()

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print('End of evaluation.')