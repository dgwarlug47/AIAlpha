import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ensemble_names = ['Chap Experiment buy and hold', 'Chap Experiment15']

ids = ['buy and hold', 'Softmax']

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

def bar_plot(y):
    sns.set_context('paper')
    sns.barplot(x = 'tickers', y = y, hue = 'id', data = final, palette = palette)
    plt.show()
    

Names_list = loadTickers(tickers_names_file_path)

def analysis(ensemble_name, id):

    all_pos = date_handler(pd.read_csv('./Ensemble/' + ensemble_name + '/ensemble position.csv'))
    all_no_transaction_cost_pnl = date_handler(pd.read_csv('./Ensemble/' + ensemble_name + '/no transaction cost pnl.csv'))
    all_transaction_cost_pnl = date_handler(pd.read_csv('./Ensemble/' + ensemble_name + '/with transaction cost pnl.csv'))


    break_even_list = []
    sharpe_list = []
    for ticker_name in Names_list:
        pos = all_pos[ticker_name]

        no_pnl = all_no_transaction_cost_pnl[ticker_name]

        no_pnl = no_pnl.fillna(0)

        with_pnl = all_transaction_cost_pnl[ticker_name]

        sharpe_list.append(with_pnl.mean()/with_pnl.std())

        np_pos = np.array(pos)

        shift_pos = copy.deepcopy(np_pos[:-1])

        shift_pos = np.append([0], shift_pos)

        turnover = np_pos - shift_pos

        turnover = np.nan_to_num(turnover, copy=True, nan=0.0, posinf=None, neginf=None)

        def break_even_calculator(t):
            return abs(sum(no_pnl - abs(t*turnover)))

        life = minimize(break_even_calculator, np.array([0]))

        break_even_list.append(life.x[0]*10000)

    stuffes = pd.DataFrame({'break_even': break_even_list, 'sharpe': sharpe_list, 
        'tickers': Names_list, 'id': id})

    return stuffes

lists_of_stuffes = []

for ensemble_name_index in range(len(ensemble_names)):
    lists_of_stuffes.append(analysis(ensemble_names[ensemble_name_index], ids[ensemble_name_index]))

final = pd.concat(lists_of_stuffes, axis=0)

#basic_bar_plot('tickers', 'sharpe', final)