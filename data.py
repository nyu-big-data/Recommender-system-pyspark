from cgi import test
import getpass
from itertools import count

from requests import head

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def partition(ratings_df):
    ratings = ratings_df.rdd
    nofRatings = ratings.count()
    nofUsers = ratings.map(lambda x: x[0]).distinct().count()
    nofMovies = ratings.map(lambda x: x[1]).distinct().count()

    print("Got %d ratings from %d users on %d movies." % (nofRatings, nofUsers, nofMovies))

    c = ratings_df.count()//2
    df1 = ratings_df.orderBy("userId", ascending=True).limit(c)
    df2 = ratings_df.orderBy("userId", ascending=False).limit(c)

    train_df1, test_df = df1.randomSplit([.6, .4])
    train_df2, val_df = df2.randomSplit([.6, .4])

    train_df =  train_df1.union(train_df2) 
    
    print(train_df.count(), val_df.count(), test_df.count())
    return train_df,val_df,test_df

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    #Small Dataset
    ratings_df = spark.read.csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    
    train_df, val_df , test_df = partition(ratings_df)
    train_df.createOrReplaceTempView('train_df')
    train_df.write.csv(f"hdfs:/user/{netID}/movielens_small_train.csv")

    val_df.createOrReplaceTempView('val_df')
    val_df.write.csv(f"hdfs:/user/{netID}/movielens_small_val.csv")

    test_df.createOrReplaceTempView('test_df')
    test_df.write.csv(f"hdfs:/user/{netID}/movielens_small_test.csv")


    #Large Dataset
    ratings_df = spark.read.csv(f'hdfs:/user/{netID}/movielens/ml-latest/ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT , timestamp INT')
    
    train_df, val_df, test_df = ratings_df.randomSplit([.6, .2, .2])
    train_df.createOrReplaceTempView('train_df')
    train_df.write.csv(f"hdfs:/user/{netID}/movielens_large_train.csv")

    val_df.createOrReplaceTempView('val_df')
    val_df.write.csv(f"hdfs:/user/{netID}/movielens_large_val.csv")

    test_df.createOrReplaceTempView('test_df')
    test_df.write.csv(f"hdfs:/user/{netID}/movielens_large_test.csv")

    




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
