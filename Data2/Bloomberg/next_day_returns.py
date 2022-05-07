import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
info = {}

info['tickers_names_file_path'] = './Bond_tickers'
info['tickers_base_path'] = './Futures Deep Momentum Features Bloomberg/'
info['num_shuffling'] = 80
info['num_trials'] = 1
info['end_to_end'] = True
info['device'] = torch.device('cuda')
info['vol_tgt'] = 0.15
info['start_date'] = '2016-01-01'
info['end_date'] = '2021-01-01'
info['label_type'] = 'standard'

def date_handler(pnl):
    pnl = pnl.rename(columns = {'Unnamed: 0': 'date'})
    new_days = list(pnl['date'])
    pnl = pnl.drop(columns= ['date'], axis=1)
    pnl.index = new_days
    return pnl

base_path = './futures'

path_for_real_returns = base_path + '.csv'
Data = pd.read_csv(path_for_real_returns)
Data = date_handler(Data)
Data = Data.pct_change().shift(-1)
Data.to_csv('next day returns.csv')