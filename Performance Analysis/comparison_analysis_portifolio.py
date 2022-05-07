import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import empyrical as ep
from scipy.optimize import minimize

ensemble_names = ['Equity_equal_weights', 'All_volatility_scaling_equal_weights', 'Equity_TPE_Top_30', 'All_TPE_Top_30']

ids = ['equal weights', 'equal weights vol scaling', 'trained only with equity', 'trained with all']

palette = 'deep'

tickers_names_file_path = '../Data/Bloomberg/tickers_Equity'

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

def bar_plot(x, y, data):
    sns.set_context('paper')

    sns.barplot(x = x, y = y, hue = 'id', data = data, palette = palette)

    plt.show()

tickers = loadTickers(tickers_names_file_path)

def results_getter(ensemble_name, id):
    new = date_handler(pd.read_csv('./Ensemble/' + ensemble_name + '/no transaction cost pnl.csv'))
    new = new[tickers]
    results = pd.DataFrame()

    portifolio = new.mean(axis=1)

    print(portifolio.mean()/portifolio.std())

    results['expected_returns'] = [portifolio.mean() * 252]
    results['vol'] = [portifolio.std() * np.sqrt(252)]
    results['downside deviation'] = [np.sqrt(252) * portifolio[portifolio < 0].std()]
    results['sharpe_ratio'] =[ep.stats.sharpe_ratio(portifolio, period='daily')]
    results['sortino'] = [ep.stats.sortino_ratio(portifolio, period='daily', annualization=None)]
    results['calmar'] = [ep.stats.calmar_ratio(portifolio, period='daily', annualization=None)]
    #results['percentage of positive returns'] = [len(portifolio[portifolio > 0])/(len(portifolio))]
    #results['avg P/avg L'] = [-((portifolio[portifolio > 0].mean())/(portifolio[portifolio < 0].mean()))]

    results.index = [id]

    return results


lists_of_results = []

for ensemble_name_index in range(len(ensemble_names)):
    lists_of_results.append(results_getter(ensemble_names[ensemble_name_index], ids[ensemble_name_index]))

final = pd.concat(lists_of_results, axis=0)
