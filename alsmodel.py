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
  
    regParam = 0.1
   
    als = ALS(maxIter=5, regParam=regParam, rank=5, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(train_df)

    # Evaluation

    #RMSE
    predictions = model.transform(val_df)
    new_predictions = predictions.filter(F.col('prediction') != np.nan)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(new_predictions)
    print ("the rmse before hyperparameter training: {}".format(rmse))
 
    #Ranking Metrics
    
    #Validation
    users = val_df.select(als.getUserCol()).distinct()
    userRecs = model.recommendForUserSubset(users,100)

    preds = userRecs.select(userRecs.userId, F.explode(userRecs.recommendations.movieId))
    prediction = preds.groupby('userId').agg(F.collect_list('col').alias("col"))

    labels = val_df.groupby('userId').agg(F.collect_list('movieId').alias("movieId"))

    predAndLabels = (prediction.join(labels, 'userId').rdd.map(lambda row: (row[1], row[2])))

    metrics = RankingMetrics(predAndLabels)
    print("Val")
    print("Mean Average Precision", metrics.meanAveragePrecision)
    print("Average Precision", metrics.precisionAt(5))
    print("NDCG", metrics.ndcgAt(5))
    print("Average Recall", metrics.recallAt(5))


    #Test
    users = test_df.select(als.getUserCol()).distinct()
    userRecs = model.recommendForUserSubset(users,100)

    preds = userRecs.select(userRecs.userId, F.explode(userRecs.recommendations.movieId))
    prediction = preds.groupby('userId').agg(F.collect_list('col').alias("col"))

    labels = test_df.groupby('userId').agg(F.collect_list('movieId').alias("movieId"))

    predAndLabels = (prediction.join(labels, 'userId').rdd.map(lambda row: (row[1], row[2])))

    metrics = RankingMetrics(predAndLabels)
    print("Test")
    print("Mean Average Precision", metrics.meanAveragePrecision)
    print("Average Precision", metrics.precisionAt(5))
    print("NDCG", metrics.ndcgAt(5))
    print("Average Recall", metrics.recallAt(5))

    
    # Hyperparameter Tuning
    # als = ALS(maxIter=5, regParam=regParam, rank=5, userCol="userId", itemCol="movieId", ratingCol="rating")
    # paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.1, 0.01]).addGrid(als.rank, range(4, 12)).build()
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # crossval = CrossValidator(estimator=als,
    #                       estimatorParamMaps=paramGrid,
    #                       evaluator=evaluator,
    #                       numFolds=5)
    # model = crossval.fit(train_df)
    # bestModel = model.bestModel
    # final_pred = bestModel.transform(val_df)
    # final_pred = final_pred.filter(F.col('prediction') != np.nan)
    # rmse = evaluator.evaluate(final_pred)
    # print ("the rmse for Validation set: {}".format(rmse))


    # final_pred = bestModel.transform(test_df)
    # final_pred = final_pred.filter(F.col('prediction') != np.nan)
    # rmse = evaluator.evaluate(final_pred)
    # print ("the rmse for Test set is: {}".format(rmse))



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