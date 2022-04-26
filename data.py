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

    # Load the boats.txt and sailors.json data into DataFrame
    movies_df = spark.read.csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/movies.csv' ,header=True)
    ratings_df = spark.read.csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')


    ratings = ratings_df.rdd
    nofRatings = ratings.count()
    nofUsers = ratings.map(lambda x: x[0]).distinct().count()
    nofMovies = ratings.map(lambda x: x[1]).distinct().count()

    print("Got %d ratings from %d users on %d movies." % (nofRatings, nofUsers, nofMovies))

    train_df, val_df, test_df = ratings_df.randomSplit([.6, .2, .2])
    print(train_df.count(), val_df.count(), test_df.count())

    train_df.createOrReplaceTempView('train_df')
    train_df.write.csv("hdfs:/user/mmk9369/movielens_train.csv")

    val_df.createOrReplaceTempView('val_df')
    val_df.write.csv("hdfs:/user/mmk9369/movielens_val.csv")

    test_df.createOrReplaceTempView('test_df')
    test_df.write.csv("hdfs:/user/mmk9369/movielens_test.csv")

    




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)