import sys
from requests import head

# And pyspark.sql to get the spark session
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import os  


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

    train_pd = train_df.toPandas()
    val_pd = val_df.toPandas()
    test_pd = test_df.toPandas()

    os.makedirs('data', exist_ok=True)
    train_pd.to_csv('data/train.csv')
    val_pd.to_csv('data/val.csv')
    test_pd.to_csv('data/test.csv')


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extentionCSV').getOrCreate()

    # Get user netID from the command line
    train = sys.argv[1]
    val = sys.argv[2]
    test = sys.argv[3]

    # Call our main routine
    main(spark, train, val, test)