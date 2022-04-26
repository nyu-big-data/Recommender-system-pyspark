import getpass
from itertools import count

from requests import head

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Lab 3 Example dataframe loading and SQL query')

    # Load the boats.txt and sailors.json data into DataFrame
    movies = spark.read.csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/movies.csv' ,header=True)
    ratings = spark.read.csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv', header=True)

    movies.show(5)
    ratings.show(5)




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)