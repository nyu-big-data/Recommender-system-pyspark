import getpass
from itertools import count

from requests import head

# And pyspark.sql to get the spark session
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''


    train_df = spark.read.csv('hdfs:/user/mmk9369/movielens_train.csv', header=True)
    val_df = spark.read.csv('hdfs:/user/mmk9369/movielens_val.csv', header=True)
    test_df = spark.read.csv('hdfs:/user/mmk9369/movielens_test.csv', header=True)
  
    regParam = 0.1
    ranks = range(4, 12)
    errors = []


    min_error = 9999

    for rank in ranks:
        als = ALS(maxIter=5, regParam=regParam, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating")
        model = als.fit(train_df)
        predictions = model.transform(val_df)
        new_predictions = predictions.filter(F.col('prediction') != np.nan)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(new_predictions)
        errors.append(rmse)

        print('For rank %s the RMSE is %s' % (rank, rmse))
        if rmse < min_error:
            min_error = rmse
            best_rank = rank
    print('The best model was trained with rank %s' % best_rank)

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10).show(5)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10).show(5)




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)