<h1>Final Project Report</h1>


<h3>Step 1: Data Partitioning</h3>
The data was first split into 2 parts to separate the data and prevent overlapping.Then the first part was split randomly in the ratio 0.8 and 0.2 into train and test.The second part was also split randomly in the ratio 0.8 and 0.2 but named train and val. Next train dataframes from both the halves were merged again giving us a train, test and validate sets.

Splitting up the data using the above way ensured that no user in test and val overlap and it cannot now know any values from val beforehand. The same splitting technique was used for the larger dataset.

<h3>Step 2: Baseline Popularity Model Implementation </h3>
Now using simple spark data frame operations top 100 highest average rated movies were identified which have atleast 50 ratings count. This technique was used because it helped in identifying the most popular movies from the whole training data and all the movies with high rating but low number of ratings were removed.

Next every user's movies were identified and compared with the top 100 popular movies. Then with the help of RankingMeterics  we computed mean average precision, average precision, NDCG and average recall of the baseline popularity model. The results are included in the github repo.


<h3>Step 3: Latent Factor Model Implementation </h3>

We have used ALS model to build a latent factor model recommender systems. For now, we have not tuned the huperparameters, 'Rank'and 'Regular Parameter'. We fit the ALS model using 'regParam: 0.1' and 'rank: 5', and derive predictions of movies for test dataset users to and compare it with actual likings. We evaluate our model using RegressionEvaluator and RankingMetrics. The results are included in the github repo. 
