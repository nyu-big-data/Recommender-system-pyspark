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
import sys



def hyperParamTune(train, val, params):
    metrics = {}
    hyperparams = params
    for rank in hyperparams['rank']:  
        r = 'Rank {}'.format(rank)
        metric = {}  
        for reg in hyperparams['regParam']:
            als = ALS(maxIter=5, regParam=reg, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating")
            model = als.fit(train)
            MAP = calMetrics(als,model,val)   
            regParam = 'Reg Param {}'.format(reg)   
            metric[regParam] = MAP
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
    metrics = metric.meanAveragePrecision
    return metrics

def main(spark, data):
    
    df = spark.read.csv(data, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')

    train_df, val_test = df.randomSplit([0.8, 0.2], seed=12345)
    test_df, val_df = val_test.randomSplit([0.5, 0.5], seed=12345)
        
    params = {"rank":[100,125,150,175,200], "regParam": [0.01, 0.1, 1, 10]}

    st = time()
    metrics = hyperParamTune(train_df, val_df, params)
    end = round(time()-st, 3)
    print("Hyperparameter tuning took {} seconds".format(end))

    maxMetric = -999
    for rank in metrics.keys():
        for regParam in metrics[rank]:
            if metrics[rank][regParam] > maxMetric:
                maxMetric = metrics[rank][regParam]
                maxRank = rank
                maxRegParam = regParam         
    bestRegParam, bestRank = float(str.split(maxRegParam, ' ')[2]), int(str.split(maxRank, ' ')[1])
    print("Best rank: {}, best reg: {}".format(bestRank, bestRegParam))

    st = time()
    als = ALS(maxIter=5, regParam=bestRegParam, rank=bestRank, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(train_df)
    end = round(time()-st, 3)
    
    testMetric = calMetrics(als, model, test_df)
    print("Evaluation on test data: {}".format(testMetric))
    print("Final model training and fitting took {} seconds".format(end))
    
    print(pd.DataFrame(metrics))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('latent factor').getOrCreate()

    # Get user netID from the command line
    data = sys.argv[1]

    # Call our main routine
    main(spark, data)