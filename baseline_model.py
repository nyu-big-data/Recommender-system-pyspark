import getpass
from itertools import count
from unittest import result
import sys
from requests import head

# And pyspark.sql to get the spark session
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


def cal_acc(labels, prediction):
    labels_df = labels.toPandas()
    pred_df = prediction.toPandas()
    uId = []
    a = []
    result_df = pd.DataFrame(columns = ['userId', 'acc'])
    users = labels_df['userId'].unique()
    print("Number of users" , len(users))
    for user in users:
        uId.append(user) 
        df1 = labels_df[labels_df['userId'] == user]
        df = pred_df.merge(df1, on='movieId', how='inner')
        acc = df.shape[0]/100
        a.append(acc)

    result_df['userId'] = uId
    result_df['acc'] = a
    return result_df

def main(spark, file_path):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    train_df = spark.read.csv(file_path, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    val_df = spark.read.csv(file_path, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    test_df = spark.read.csv(file_path, header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')

    predictions = train_df.groupBy(train_df.movieId).agg(F.mean('rating').alias('AvgRating'), F.count('rating').alias("count")).filter(F.col('count')>=50).orderBy("AvgRating", ascending = False).limit(100)
    
    val_window = Window.partitionBy(val_df['userId']).orderBy(val_df['rating'].desc())
    valLabels = val_df.select('*', F.row_number().over(val_window).alias('counts')).filter(F.col('counts') <= 100)

    result = cal_acc(valLabels,predictions)
    result_df = spark.createDataFrame(result)
    result_df.write.csv("val_output.csv")

    test_window = Window.partitionBy(test_df['userId']).orderBy(test_df['rating'].desc())
    testLabels = test_df.select('*', F.row_number().over(test_window).alias('counts')).filter(F.col('counts') <= 100)

    result = cal_acc(testLabels,predictions)
    result_df = spark.createDataFrame(result)
    result_df.write.csv("test_output.csv")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    file_path = sys.argv[1]

    # Call our main routine
    main(spark, file_path)