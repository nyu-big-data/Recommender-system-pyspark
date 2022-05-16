import numpy as np
import pandas as pd
from lightfm import LightFM
import itertools
from lightfm.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from time import time
from lightfm.cross_validation import random_train_test_split

def hyperParamTune(train, val, params, m_iter):
    metrics = {}
    hyperparams = params
    for rank in hyperparams['rank']:  
        r = 'Rank {}'.format(rank)
        metric = {}  
        for reg in hyperparams['regParam']:
            model = LightFM(loss = 'warp', user_alpha = reg, no_components = rank)
            model.fit(train, epochs=m_iter, verbose=False)
            MAP = precision_at_k(model, val, k = 100).mean()   
            regParam = 'Reg Param {}'.format(reg)   
            metric[regParam] = MAP
        metrics[r] = metric          
    return metrics  


def main():
    
    df = pd.read_csv('/scratch/mmk9369/movielens/ml-latest-small/ratings.csv')
    print(len(df['userId'].unique()), len(df['movieId'].unique()), df.shape)

    # dfNew = df.sample(frac=0.5)
    # print(len(dfNew['userId'].unique()), len(dfNew['movieId'].unique()), dfNew.shape)
    
    df_interaction = pd.pivot_table(df, index='userId', columns='movieId', values='rating')
    print(df_interaction.shape)
    
    dfMatrix = csr_matrix(df_interaction.values)
    dfMatrix.data = np.nan_to_num(dfMatrix.data, copy=False)
    print(dfMatrix.shape)

    (train, valTest) = random_train_test_split(dfMatrix, test_percentage=0.8)
    (val, test) = random_train_test_split(valTest, test_percentage=0.5)
    print(train.shape, val.shape, test.shape)

    params = {"rank": [100,125,150,175,200], "regParam": [0.01, 0.1, 1, 10]}
    
    #Hyperparameter Tuning
    st = time()
    metrics = hyperParamTune(train, val, params, m_iter = 10)
    end = round(time()-st, 3)
    print("Hyperparameter tuning took {} seconds".format(end))

    #Best Parameters
    maxMetric = -999
    for rank in metrics.keys():
        for regParam in metrics[rank]:
            if metrics[rank][regParam] > maxMetric:
                maxMetric = metrics[rank][regParam]
                maxRank = rank
                maxRegParam = regParam         
    bestRegParam, bestRank = float(str.split(maxRegParam, ' ')[2]), int(str.split(maxRank, ' ')[1])
    print("Best rank: {}, best reg: {}".format(bestRank, bestRegParam))

    #Model Fitting
    st = time()
    model = LightFM(loss = 'warp', user_alpha = bestRegParam, no_components = bestRank)
    model.fit(train, epochs=10, verbose=False)
    end = round(time()-st, 3)
    
    metric =  precision_at_k(model, test, k=100).mean()
    print("Evaluation on test data: {}".format(metric))
    print("Final model training and fitting took {} seconds".format(end))
    
    print(pd.DataFrame(metrics))

main()
