<h1>Final Project Report</h1>

<h2>Introduction</h2>
lorem ipsum:-

<h3>Step 1: Data Partitioning</h3>
The data was first split into 2 parts to separate the data and prevent overlapping.Then the first part was split randomly in the ratio 0.8 and 0.2 into train and test.The second part was also split randomly in the ratio 0.8 and 0.2 but named train and val. Next train dataframes from both the halves were merged again giving us a train, test and validate sets.

Splitting up the data using the above way ensured that no user in test and val overlap and it cannot now know any values from val beforehand. The same splitting technique was used for the larger dataset.

<h3>Step 2: Baseline Popularity Model Implementation </h3>
Now using simple spark data frame operations top 100 highest average rated movies were identified which have atleast 50 ratings count. This technique was used because it helped in identifying the most popular movies from the whole training data and all the movies with high rating but low number of ratings were removed.

Next every user's movies were identified and compared with the top 100 popular movies. Then with the help of RankingMeterics  we computed mean average precision, average precision, NDCG and average recall of the baseline popularity model. The results are included in the github repo.


<h3>Step 3: Latent Factor Model Implementation </h3>

We have used ALS model to build a latent factor model recommender systems. For now, we have not tuned the huperparameters, *Rank* and *Regular Parameter*. We fit the ALS model using *regParam: 0.1* and *rank: 5*, and derive predictions of movies for test dataset users to and compare it with actual likings. We evaluate our model using RegressionEvaluator and RankingMetrics. The results are included in the github repo.

<h2>Evaluation</h2>
The performance of our model was evaluated using Mean Average Precision(MAP) as it measures how many recommended titles are present in the user's own viewed titles where the order of recommendations are also considered[MAP](https://spark.apache.org/docs/1.5.0/mllib-evaluation-metrics.html).

We used Spark's built in MAP function which calculates similarity of the user's top rated titles with the recommended top rated titles and gives the precision value.

<h3>Hyperparameter tuning</h3>
Before running ALS model on full dataset we created a dictionary for various values of regParam to be tried with each value of Rank which is the number of latent factors considered. We observed increase in MAP values for increase in number of latent factors up to some point. 

<h3>Result</h3>
After running the ALS model over multiple epochs we calculated various test metrics like RMSE, p, ndcg and recall. The best results were obtained when rank =200, regParam = 0.1 and MAP value obtained was 0.0878741265074142365. 

Results for small dataset:-
(i) Before hyperparameter training
|Iteration  |rmse   |MAP   |p   |ndcg   |recall   |
|---|---|---|---|---|---|
| 10  |0.9048930872645341   |0.00978364643024941   |0.03972307692307692   |0.07236193227820072   |0.10302455581625031   |
|15|0.9208599048883793|0.008719369859523051|0.03529230769230771|0.06736585084566271|0.10179554157618477|

(ii)After starting hyperparameter tuning 
|Iteration |rmse   |MAP   |p   |ndcg   |recall   |
|---|---|---|---|---|---|
| 10  |1.492005099702937   |0.019533966626636874   |0.052061538461538485   |0.11108940889217304   |0.17476783194314033 |
|15|1.408975102523959|0.02320768899588743|0.05193846153846156|0.11713272557876755|0.17238921247522646|

The best rank was 200 and the best regParam was 0.01

Results for large dataset:-
(i) Before hyperparameter training
|Iteration  |rmse   |MAP   |p   |ndcg   |recall   |
|---|---|---|---|---|---|
| 10  |0.7817637332880522   |0.0011426337166653446   |0.0006430182192932087 |0.0050636760057387814 |0.01751698369389073   |
|15|0.781763733227067|0.0011423946066489875|0.0006430562474568668|0.005063116065795393|0.01751701387497299|

(ii)After starting hyperparameter tuning 
|Iteration |rmse   |MAP   |p   |ndcg   |recall   |
|---|---|---|---|---|---|
| 10  |0.6911283941016878   |0.08623654270210764   |0.016303662492441907   |0.1620966543949737   |0.2356826709672368 |
|15|0.692514769810245|0.0878741265074142365|0.0175820147365894147|0.182546987118876664|0.24258487115186161|

The best rank was 200 and the best regParam was 0.01


