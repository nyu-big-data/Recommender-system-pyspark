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

def hyperParamTune(train_df):

    als = ALS(maxIter=5, regParam=0.1, rank=100, userCol="userId", itemCol="movieId", ratingCol="rating")
    paramGrid = ParamGridBuilder().addGrid(als.rank, [100,120,140,160,180,200]).addGrid(als.regParam, [0.01, 0.1, 1, 10]).build()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    crossval = CrossValidator(estimator=als,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=5)
    model = crossval.fit(train_df)
    bestModel = model.bestModel
    print("**Best Model**")
    print("  Rank:", bestModel._java_obj.parent().getRank())
    print("  RegParam:", bestModel._java_obj.parent().getAlpha())
    return als, bestModel

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
    metrics['p'] = metric.precisionAt(5)
    metrics['ndcg'] = metric.ndcgAt(5)
    metrics['recall'] = metric.recallAt(5)
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
  
   
    als = ALS(maxIter=5, regParam=0.1, rank=100, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(train_df)

    print("Starting Evaluation")
    # Evaluation
    valMetrics = calMetrics(als ,model, val_df)
    testMetrics = calMetrics(als, model, test_df)

    print("Validation Metrics", valMetrics)
    print("Test Metrics", testMetrics)

    print("Starting HyperParameter Tuning")
    # Hyperparameter Tuning
    newAls, newModel = hyperParamTune(train_df)

    valMetrics = calMetrics(newAls ,newModel, val_df)
    testMetrics = calMetrics(newAls, newModel, test_df)

    print("Validation Metrics", valMetrics)
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