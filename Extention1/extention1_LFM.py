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
            model = LightFM(random_state = 123, learning_rate = reg, no_components = rank)
            model.fit(train, epochs=m_iter)
            MAP = precision_at_k(model, val, k = 100).mean()   
            regParam = 'Reg Param {}'.format(reg)   
            metric[regParam] = MAP
        metrics[r] = metric          
    return metrics  


def main():
    
    train_df = pd.read_csv('/scratch/mmk9369/movielens_small_train.csv')
    test_df = pd.read_csv('/scratch/mmk9369/movielens_small_test.csv')
    val_df = pd.read_csv('/scratch/mmk9369/movielens_small_val.csv')
    
    df = pd.concat([train_df, val_df, test_df])
    dfNew = pd.pivot_table(df, index='userId', columns='movieId', values='rating')

    df_matrix = csr_matrix(dfNew.values)
    trainMatrix, valTestMatrix = random_train_test_split(df_matrix, test_percentage=0.8)
    valMatrix, testMatrix = random_train_test_split(valTestMatrix, test_percentage=0.5)

    params = {"rank": [100,120,140,160,180,200], "regParam": [0.01, 0.1, 1, 10]}
    
    st = time()
    metrics = hyperParamTune(trainMatrix, valMatrix, params, m_iter = 4)
    end = round(time()-st, 3)
    print("Hyperparameter tuning took {} seconds".format(end))


    maxMetric = 0
    for rank in metrics.keys():
        for regParam in metrics[rank]:
            if metrics[rank][regParam] > maxMetric:
                maxRank = rank
                maxRegParam = regParam         
    bestRegParam, bestRank = float(str.split(maxRegParam, ' ')[0]), int(str.split(maxRank, ' ')[0])
    print("Best rank: {}, best reg: {}".format(bestRank, bestRegParam))

    st = time()
    model = LightFM(random_state = 123, learning_rate = bestRegParam, no_components = bestRank)
    model = model.fit(train_df, epochs = 5)
    metric =  precision_at_k(model, test_df).mean()
    end = round(time()-st, 3)
    
    print("Evaluation on test data: {}".format(metric))
    print("Final model training and fitting took {}".format(end))
    
    return pd.DataFrame(metrics)

main()