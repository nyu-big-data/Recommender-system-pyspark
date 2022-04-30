import getpass
from itertools import count
from unittest import result
import sys
from requests import head

# And pyspark.sql to get the spark session
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics


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

    predictions = train_df.groupBy(train_df.movieId).agg(F.mean('rating').alias('AvgRating'), F.count('rating').alias("count")).filter(F.col('count')>=50).orderBy("AvgRating", ascending = False).limit(100)  

    labels = test_df.groupby('userId').agg(F.collect_list('movieId').alias("movieId"))
    
    sc = SparkContext
    predictions_df = predictions.toPandas()
    labels_df = labels.toPandas()
    pred = predictions_df['movieId'].tolist()
    predAndLabel = [(pred, x['movieId']) for x in labels_df.iterrows()]
    predAndLabels = sc.parallelize(predAndLabel)
    
    metrics = RankingMetrics(predAndLabels)
    print("Mean Average Precision", metrics.meanAveragePrecision)
    print("Average Precision", metrics.precisionAt(5))
    print("NDCG", metrics.ndcgAt(5))
    print("Average Recall", metrics.recallAt(5))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline popularity').getOrCreate()

    # Get user netID from the command line
    train = sys.argv[1]
    val = sys.argv[2]
    test = sys.argv[3]

    # Call our main routine
    main(spark, train, val, test)