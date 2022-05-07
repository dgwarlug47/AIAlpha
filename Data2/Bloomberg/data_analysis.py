import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
# this script is basically to compute basic information about the data you are dealing with
# for now the only features we have are the stds and the amount of nans

tickers_names_file_path = './tickers_FX'
tickers_base_path = './all_futures.csv'
start_date = '2000-01-01'
end_date = '2019-01-01'
playground = True
quiet = False

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

if not playground:
    if not os.path.exists('./basic_data_analysis'):
        os.makedirs('./basic_data_analysis')

    if not os.path.exists('./basic_data_analysis/' + tickers_names_file_path):
        os.makedirs('./basic_data_analysis/' + tickers_names_file_path)

prices = date_handler(pd.read_csv(tickers_base_path))
prices = prices[prices.index > start_date]
prices = prices[prices.index < end_date]
list_nan_beginings = []
list_stds = []

list_tickers = loadTickers(tickers_names_file_path)
prices = prices[list_tickers]


all_returns = prices.pct_change()

cumprod = (1.+all_returns).rolling(window=21*3).agg(lambda x : x.prod())

fig = sns.heatmap(cumprod.corr(), annot=True)
if not playground:
    plt.savefig('./basic_data_analysis/' + tickers_names_file_path + '/correlation')
else:
    plt.show()

fig.clear()

for ticker_index in all_returns.columns:
    #non_zeros_returns_index = (np.nonzero(data[ticker_index, :, 0]))[0]
    #list_nan_beginings.append(np.min(non_zeros_returns_index)/len(info['all_days']))
    list_stds.append((all_returns[ticker_index]).std())

#nan = pd.DataFrame({'nan': list_nan_beginings, 'tickers': list_tickers})
stds = pd.DataFrame({'std': list_stds, 'tickers': list_tickers})

def basic_bar_plot(x, y, data):
    sns.set_context('paper')
    sns.barplot(x = x, y = y, data = data,
                palette = 'PuRd',ci=None 
                )
    if not playground:
        plt.savefig('./basic_data_analysis/' + tickers_names_file_path + '/stds')
    else:
        plt.show()

basic_bar_plot(x='std', y='tickers', data=stds)