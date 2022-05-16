import getpass
from itertools import count
import sys
from requests import head

# And pyspark.sql to get the spark session
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def hyperParamTune(train, val, params):
    #Ranking Metrics Evaluator
    metrics = {}
    hyperparams = params
    for rank in hyperparams['rank']:  
        r = 'Rank {}'.format(rank)
        metric = {}  
        for reg in hyperparams['regParam']:
            als = ALS(maxIter=15, regParam=reg, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
            model = als.fit(train)
            rankMetrics = calMetrics(als,model,val)   
            regParam = 'Reg Param {}'.format(reg)   
            metric[regParam] = rankMetrics
        metrics[r] = metric          
    return metrics

    #RMSE Evaluator
    # als = ALS(maxIter=20, regParam=0.1, rank=100, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    # paramGrid = ParamGridBuilder().addGrid(als.rank, [100,120,140,160,180,200]).addGrid(als.regParam, [0.01, 0.1, 1, 10]).build()
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # crossval = CrossValidator(estimator=als,
    #                         estimatorParamMaps=paramGrid,
    #                         evaluator=evaluator,
    #                         numFolds=5)
    # model = crossval.fit(train_df)
    # bestModel = model.bestModel
    # print("**Best Model**")
    # print("  Rank:", bestModel._java_obj.parent().getRank())
    # print("  RegParam:", bestModel._java_obj.parent().getAlpha())
    # return als, bestModel

def bestParams(metrics):
    maxMAP = maxP = maxNDCG = maxRecall = -999
    for rank in metrics.keys():
        for regParam in metrics[rank]:
            if metrics[rank][regParam]['MAP'] > maxMAP and metrics[rank][regParam]['p'] > maxP and metrics[rank][regParam]['ndcg'] > maxNDCG and metrics[rank][regParam]['recall'] > maxRecall:

                maxMAP = metrics[rank][regParam]['MAP']
                maxP = metrics[rank][regParam]['p']
                maxNDCG = metrics[rank][regParam]['ndcg']
                maxRecall = metrics[rank][regParam]['recall']
                maxRank = rank
                maxRegParam = regParam      

    return float(str.split(maxRegParam, ' ')[2]), int(str.split(maxRank, ' ')[1])


def calMetrics(als, model, df):
    metrics = {}
    #RMSE
    predictions = model.transform(df)
    new_predictions = predictions.filter(F.col('prediction') != np.nan)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(new_predictions)
    metrics['rmse'] = rmse
 
    #Ranking Metrics
    users = df.select(als.getUserCol()).distinct()
    userRecs = model.recommendForUserSubset(users,100)

    preds = userRecs.select(userRecs.userId, F.explode(userRecs.recommendations.movieId))
    prediction = preds.groupby('userId').agg(F.collect_list('col').alias("col"))

    labels = df.groupby('userId').agg(F.collect_list('movieId').alias("movieId"))

    predAndLabels = (prediction.join(labels, 'userId').rdd.map(lambda row: (row[1], row[2])))

    metric = RankingMetrics(predAndLabels)
    metrics['MAP'] = metric.meanAveragePrecision
    metrics['p'] = metric.precisionAt(100)
    metrics['ndcg'] = metric.ndcgAt(100)
    metrics['recall'] = metric.recallAt(100)
    return metrics
 

def main(spark, train, val, test):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    train_df = spark.read.csv(train, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    val_df = spark.read.csv(val, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    test_df = spark.read.csv(test, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
  
   
    als = ALS(maxIter=15, regParam=0.1, rank=100, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_df)

    print("Starting Evaluation")
    # Evaluation
    testMetrics = calMetrics(als, model, test_df)
    print("Test Metrics", testMetrics)

    print("Starting HyperParameter Tuning")
    # Hyperparameter Tuning

    params = {"rank":[100,125,150,175,200], "regParam": [0.01, 0.1, 1, 10]}
    metrics = hyperParamTune(train_df, val_df, params)

    bestRegParam, bestRank = bestParams(metrics)
    print("Best rank: {}, best reg: {}".format(bestRank, bestRegParam))

    newAls = als = ALS(maxIter=15, regParam=bestRegParam, rank=bestRank, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    newModel = newAls.fit(train_df)
    testMetrics = calMetrics(newAls, newModel, test_df)
    print("Test Metrics", testMetrics)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('latent factor').getOrCreate()

    # Get user netID from the command line
    train = sys.argv[1]
    val = sys.argv[2]
    test = sys.argv[3]

    # Call our main routine
    main(spark, train, val, test)