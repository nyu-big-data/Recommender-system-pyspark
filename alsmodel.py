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
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''


    train_df = spark.read.csv(f'hdfs:/user/{netID}/movielens_train.csv', header=True,schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    val_df = spark.read.csv(f'hdfs:/user/{netID}/movielens_val.csv', header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    test_df = spark.read.csv(f'hdfs:/user/{netID}/movielens_test.csv', header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
  
    regParam = 0.1
   
    als = ALS(maxIter=5, regParam=regParam, rank=5, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(train_df)
    predictions = model.transform(val_df)
    new_predictions = predictions.filter(F.col('prediction') != np.nan)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(new_predictions)
    print ("the rmse : {}".format(rmse))
 
    # # Generate top 10 movie recommendations for each user
    # userRecs = model.recommendForAllUsers(10).select('userId','recommendations').show(1)
    # # Generate top 10 user recommendations for each movie
    # movieRecs = model.recommendForAllItems(10).select('movieId', 'recommendations').show(1)
        # Hyperparameter Tuning

    
    als = ALS(maxIter=5, regParam=regParam, rank=5, userCol="userId", itemCol="movieId", ratingCol="rating")
    paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.1, 0.01]).addGrid(als.rank, range(4, 12)).build()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)
    model = crossval.fit(train_df)
    bestModel = model.bestModel
    final_pred = bestModel.transform(val_df)
    final_pred = final_pred.filter(F.col('prediction') != np.nan)
    rmse = evaluator.evaluate(final_pred)
    print ("the rmse for Validation set: {}".format(rmse))


    final_pred = bestModel.transform(test_df)
    final_pred = final_pred.filter(F.col('prediction') != np.nan)
    rmse = evaluator.evaluate(final_pred)
    print ("the rmse for Test set is: {}".format(rmse))



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)