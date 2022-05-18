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

<h3>Single Machine Implementation</h3>
In this section, we introduce lightfm package for recommendation system implementation in Python and compare our experiment results. We also utilized pandas and SciPy in intermediate steps for data structure transformation.

<h4>Implementation of LightFM packages</h4>
lightfm is a package designed for various types of recommender systems. It covers functions for both implicit and explicit feedback models, and can be applied to collaborative filtering models, hybrid models as well as cold-start recommendations.
In this project we only explored the collaborative filtering part of the lightfm package. A summary of implementation steps is listed as follows:
● Input dataframe of interactions, convert to initial utility matrix with pandas.pivot_table, then convert to coordinate format with scipy.sparse.crs_matrix which is the acceptable format for fitting lightfm models.
● Training and test split with built-in function lightfm.cross_validation. We set split ratio as 0.8/0.2 and split into train and test set, corresponding with data splitting method for Spark ALS model.
● Fit dataset into a lightfm instance, with corresponding hyperparameters selected from Section 6. We conducted experiments on Weighted Approximate Rank Pairwise Loss Personalized Ranking for better performance comparison. The time for model fitting was noted and then compared with ALS model.
● Get test score of precision at k = 100 and takemean of it. And compared this with ALS MAP Ranking metric.

<h4>Performance comparison with Spark’s ALS model</h4>
We ran lightfm model on 50% (50418,4) and 100% (100836, 4) of small dataset, with the same hyperparameter combinations as in Spark ALS. Some instant observations are:
● LightFM in comparison with ALS model, gives better performance in terms of both the accuracy and model fitting time, on same dataset and hyper parameters.
● The optimal hyperparameter combinations for Spark ALS and LightFM do not agree. This may result from different splitting methodologies of original dataset
● LightFM and ALS when compared on different dataset sizes produces same trend i.e., LightFM is faster than ALS, but when We tried to run a huge dataset like the Movielens large dataset on both the models. LightFM gave us the error of memory full and were not able to predict movie recommendations whereas ALS model was easily able to process the large dataset. Hence,LightFM is better on small datasets, but we cannot implement this model on huge datasets (which is usually the case in most recommender systems problems). Also, lightfm only contains limited built-in evaluation metrics. So, this makes ALS preferable when dealing with large datasets.


(i)Whole small dataset on LightFM
| RegPara m | Rank 100 | Rank 125 | Rank 150 | Rank 175 | Rank 200 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.01 | 0.386197 | 0.385590 | 0.384410 | 0.386049 | 0.388393 |
| 0.1 | 0.397918 | 0.398607 | 0.398902 | 0.398328 | 0.399000 |
| 1 | 0.397820 | 0.397049 | 0.398869 | 0.399902 | 0.392656 |
| 10 | 0.400918 | 0.400393 | 0.400852 | 0.401115 | 0.400574 |

(ii)50% small dataset on LightFM
| RegPar am | Rank 100 | Rank 125 | Rank 150 | Rank 175 | Rank 200 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.01 | 0.383607 | 0.383180 | 0.382738 | 0.380623 | 0.382344 |
| 0.1 | 0.399131 | 0.395410 | 0.397344 | 0.398836 | 0.396541 |
| 1 | 0.397754 | 0.395295 | 0.395115 | 0.398656 | 0.399656 |
| 10 | 0.396902 | 0.397803 | 0.398180 | 0.395918 | 0.396934 |

(iii)Whole small dataset on ALS
| RegPara m | Rank 100 | Rank 125 | Rank 150 | Rank 175 | Rank 200 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.01 | 0.013289 | 0.013139 | 0.013796 | 0.013589 | 0.014587 |
| 0.1 | 0.005627 | 0.005709 | 0.005709 | 0.005849 | 0.005758 |
| 1 | 0.000009 | 0.000009 | 0.000009 | 0.000009 | 0.000009 |
| 10 | 0.000004 | 0.000004 | 0.000004 | 0.000004 | 0.000004 |

(iv) 50% small dataset on ALS
| RegPara m | Rank 100 | Rank 125 | Rank 150 | Rank 175 | Rank 200 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.01 | 0.013638 | 0.012732 | 0.013223 | 0.012420 | 0.013303 |
| 0.1 | 0.006451 | 0.006454 | 0.006594 | 0.006539 | 0.006714 |
| 1 | 0.000006 | 0.000006 | 0.000006 | 0.000006 | 0.000006 |
| 10 | 0.000048 | 0.000049 | 0.000047 | 0.000049 | 0.000047 |
