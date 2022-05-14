import numpy as np
import pandas as pd
from lightfm import LightFM
import itertools
from lightfm.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from time import time
from lightfm.cross_validation import random_train_test_split
from lightfm.datasets import fetch_movielens

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
    
    movielens = fetch_movielens()
    totalTrain, test = movielens['train'], movielens['test']
    
    train, val = random_train_test_split(totalTrain, test_percentage=0.7)

    params = {"rank": [100,120,140,160,180,200], "regParam": [0.01, 0.1, 1, 10]}
    
    st = time()
    metrics = hyperParamTune(train, val, params, m_iter = 4)
    end = round(time()-st, 3)
    print("Hyperparameter tuning took {} seconds".format(end))

    maxMetric = 0
    for rank in metrics.keys():
        for regParam in metrics[rank]:
            if metrics[rank][regParam] > maxMetric:
                maxMetric = metrics[rank][regParam]
                maxRank = rank
                maxRegParam = regParam         
    bestRegParam, bestRank = float(str.split(maxRegParam, ' ')[2]), int(str.split(maxRank, ' ')[1])
    print("Best rank: {}, best reg: {}".format(bestRank, bestRegParam))

    st = time()
    model = LightFM(random_state = 123, learning_rate = bestRegParam, no_components = bestRank)
    model = model.fit(train, epochs = 5)
    metric =  precision_at_k(model, test).mean()
    end = round(time()-st, 3)
    
    print("Evaluation on test data: {}".format(metric))
    print("Final model training and fitting took {}".format(end))
    
    return pd.DataFrame(metrics)

df = main()
df.to_csv('LFM.csv')
