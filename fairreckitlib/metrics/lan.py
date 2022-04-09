import json
from datetime import datetime

import pandas as pd
import enum

from fairreckitlib.metrics.evaluator2 import EvaluatorLenskit, EvaluatorRexmex
from fairreckitlib.metrics.metrics2 import Metric


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

test1B_rec = Test(
    recsPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/Implicit_AlternatingLeastSquares_0/ratings.tsv'
    , trainPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/train_set.tsv'
    , testPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-1B_0/test_set.tsv'
)

test360_rec = Test(
    recsPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/Implicit_AlternatingLeastSquares_0/ratings.tsv'
    , trainPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/train_set.tsv'
    , testPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/test_set.tsv'
)

test360_pred = Test(
    recsPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/LensKit_PopScore_0/ratings.tsv'
    , trainPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/train_set.tsv'
    , testPath='D:/Downloads/1649162862_HelloFRK/1649162862_HelloFRK/run_0/LFM-360K_0/test_set.tsv'
)

test_result = test360_rec
#metrics = [Metric.ndcg, Metric.precision, Metric.recall, Metric.mrr, Metric.rmse, Metric.mae]
#metrics = [Metric.ndcg]
metrics = [Metric.item_coverage]

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print('Starting evaluation..')

evaluations = []
for metric in metrics:
    if metric in EvaluatorLenskit.metricDict.keys():
        print('Lenskit:')
        evaluator = EvaluatorLenskit(train_path=test_result.trainPath,
                                     test_path=test_result.testPath,
                                     recs_path=test_result.recsPath,
                                     metrics=[metric])
        evaluations.append({str(metric): evaluator.evaluate()})
    else:
        print('Rexmex:')
        evaluator = EvaluatorRexmex(train_path=test_result.trainPath,
                                    test_path=test_result.testPath,
                                    recs_path=test_result.recsPath,
                                    metrics=[metric])
        evaluations.append({str(metric): evaluator.evaluate()})

print(evaluations)

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print('End of evaluation.')

print('Writing evaluations to file..')
out_file = open("evaluations.json", "w")
json.dump({'evaluations': evaluations}, out_file, indent=4)
# TODO: write entry in the overview JSON
print('Saved evaluations.')