import numpy as np
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from catch22_master.wrap_Python.catch22.catch22 import catch22_all

info = {}
info['tickers_names_file_path'] = './tickers_All_but_FX'
info['start_date'] = '2016-01-01'
info['end_date'] = '2021-01-01'
info['vol_tgt'] = 0.15
n_clusters = 10

def loadTickers(fileName):
    # receives as input the address where the names of the tickers are stored.
    # outputs the list of the tickers names.
    lineList = list()
    with open(fileName) as f:
        for line in f:
            lineList.append(line)
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList

def date_handler(pnl):
    pnl = pnl.rename(columns = {'Unnamed: 0': 'date'})
    new_days = list(pnl['date'])
    pnl = pnl.drop(columns= ['date'], axis=1)
    pnl.index = new_days
    return pnl

tickers = loadTickers(info['tickers_names_file_path'])

base_path = './futures'

path_for_real_returns = base_path + '.csv'
prices = pd.read_csv(path_for_real_returns)
prices = date_handler(prices)
prices = prices[tickers]
returns = prices.pct_change()
leverage_factor = (info['vol_tgt']/(np.sqrt(252)*returns.ewm(span=60).std())).shift(1)
leverage_factor = leverage_factor[leverage_factor.index <= info['end_date']]
leverage_factor = leverage_factor[leverage_factor.index >= info['start_date']]
prices = prices[prices.index <= info['end_date']]
prices = prices[prices.index >= info['start_date']]
returns = returns[returns.index <= info['end_date']]
returns = returns[returns.index >= info['start_date']]

X_array = []
for ticker in tickers:
    X_array.append(catch22_all(np.array( (returns[ticker]*leverage_factor[ticker]).dropna()))['values'])
X_array = np.array(X_array)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_array)

for cluster in range(n_clusters):
    new_cluster = []
    for y in range(len(kmeans.labels_)):
        if kmeans.labels_[y] == cluster:
            new_cluster.append(tickers[y])
    print(cluster, ':', new_cluster)