import numpy as np
import pandas as pd
from sklearn import metrics
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics
from time import time



def hyperParamTune(train, val, params):

    metrics = {}
    hyperparams = params
    for rank in hyperparams['rank']:  
        r = 'Rank {}'.format(rank)
        metric = {}  
        for reg in hyperparams['regParam']:
            als = ALS(maxIter=5, regParam=reg, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating")
            model = als.fit(train)
            pAt = calMetrics(als,model,val)   
            regParam = 'Reg Param {}'.format(reg)   
            metric[regParam] = pAt
        metrics[r] = metric          
    return metrics 

def calMetrics(als, model, df):
    metrics = 0
    #Ranking Metrics
    users = df.select(als.getUserCol()).distinct()
    userRecs = model.recommendForUserSubset(users,100)

    preds = userRecs.select(userRecs.userId, F.explode(userRecs.recommendations.movieId))
    prediction = preds.groupby('userId').agg(F.collect_list('col').alias("col"))

    labels = df.groupby('userId').agg(F.collect_list('movieId').alias("movieId"))

    predAndLabels = (prediction.join(labels, 'userId').rdd.map(lambda row: (row[1], row[2])))

    metric = RankingMetrics(predAndLabels)
    metrics = metric.precisionAt(100)
    return metrics

def main():
    
    train_df = pd.read_csv('/scratch/mmk9369/movilens_small_train.csv')
    test_df = pd.read_csv('/scratch/mmk9369/movilens_small_test.csv')
    val_df = pd.read_csv('/scratch/mmk9369/movilens_small_val.csv')
    
    params = {"rank": [100,120,140,160,180,200], "regParam": [0.01, 0.1, 1, 10]}

    st = time()
    metrics = hyperParamTune(train_df, val_df, params)
    end = round(time()-st, 3)
    print("Hyperparameter tuning took {} seconds".format(end))

    maxMetric = 0
    for rank in metrics.keys():
        for regParam in metrics[rank]:
            if metrics[rank][regParam] > maxMetric:
                maxRank = rank
                maxRegParam = regParam         
    bestRegParam, bestRank = float(str.split(maxRegParam, ' ')[0]), int(str.split(maxRank, ' ')[0])
    print("Best rank: {}, best reg: {}".format(bestRank, bestRegParam))

    st = time()
    als = ALS(maxIter=5, regParam=bestRegParam, rank=bestRank, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(train_df)
    testMetric = calMetrics(als, model, test_df)
    end = round(time()-st, 3)
    
    print("Evaluation on test data: {}".format(testMetric))
    print("Final model training and fitting took {}".format(end))
    
    return pd.DataFrame(metrics)