import pandas as pd
import numpy as np
import pyfolio as pf
import matplotlib.pyplot as plt
import datetime
import pickle
import yaml
use_bbg = True
if use_bbg:
    from bbgclient import SyncClient
    import pdblp
    con = pdblp.BCon(timeout=500000, debug = False)
    con.start()
    bbg = SyncClient(False, False)
from datetime import date 
plt.rcParams['figure.figsize'] = 14, 10
plt.rcParams['figure.figsize'] = 14, 10
import warnings
warnings.filterwarnings("ignore")

def roll_ticker(bbg_ticker,roll_type, adjust_type, days_exp):
    dict_adjust = {'None': 'N', 'Difference':'D', 'Ratio': 'R', 'Average': 'W'}
    if len(roll_type) == 1:
        lst = bbg_ticker.split()
        bbg_ticker_aux = ' '.join([w for w in lst if w != lst[-1]])
        if days_exp<10:
            return bbg_ticker_aux + ' ' + roll_type + ':' + '0'+ str(days_exp) + '_' + '0' + '_' +dict_adjust[adjust_type] +' '+ lst[-1]
        else:
            return bbg_ticker_aux + ' ' + roll_type + ':' + str(days_exp) + '_' + '0' + '_' +dict_adjust[adjust_type] +' '+ lst[-1]
    else:
        dict_rolls = {'Bloomberg Default':'B', 
                'Relative to Expiration' : 'R',
                'Fixed Day of Month': 'F',
                'With Active Future': 'A',
                'Relative to First Notice': 'N',
                'At First Delivery': 'D',
                'At option expiration':'O'}        
        lst = bbg_ticker.split()
        bbg_ticker_aux = ' '.join([w for w in lst if w != lst[-1]])
        if days_exp<10:
            return bbg_ticker_aux + ' ' + dict_rolls[roll_type] + ':' + '0'+ str(days_exp) + '_' + '0' +'_' + dict_adjust[adjust_type] +' '+ lst[-1]
        else:
            return bbg_ticker_aux + ' ' + dict_rolls[roll_type] + ':' + str(days_exp) + '_' + '0' +'_' + dict_adjust[adjust_type] +' '+ lst[-1]

def bbg_data(list_data, list_fields, start_date, end_date):
    data_bbg =  con.bdh(list_data, 
                        list_fields, 
                        start_date, 
                        end_date)
#                         elms=[("calendarCodeOverride", "5D")])
    return data_bbg

def bbg_to_df(df_bbg, dict_aux, cumreturn=True):
    columns_df = [dict_aux[i[0]] for i in df_bbg.columns]
    if cumreturn:
        df_return = ((pd.DataFrame(df_bbg.values, index = df_bbg.index, columns = columns_df)).pct_change().fillna(0) + 1).cumprod()
        return df_return
    else:
        df_ = pd.DataFrame(df_bbg.values, index = df_bbg.index, columns = columns_df)
        return df_    

import pdb
pdb.set_trace()
all_assets = pd.read_excel('assets.xlsx', engine = 'openpyxl')
futures = all_assets.dropna()
futures_tickers = []
for f in futures.index:
    futures_tickers.append(roll_ticker(futures.loc[f].Ticker, futures.loc[f].RollType, futures.loc[f].AdjustType, int(futures.loc[f].Days_exp)))
otc = all_assets[~all_assets.index.isin(futures.index)].dropna(axis=1)
otc_tickers = otc.Ticker.tolist()

dict_aux_fut = {x:y for x,y in zip(futures_tickers, futures.Name)}
dict_aux_oct = {x:y for x,y in zip(otc_tickers, [i.split()[0] for i in otc_tickers])}

date_parser = lambda x: x.strftime('%Y%m%d')
start_date = "19870102"
# start_date = "20190402"
today = date.today()
end_date = date_parser(today)

futures_data = bbg_to_df(bbg_data(futures_tickers, ['CONTRACT_VALUE'], start_date, end_date), dict_aux_fut, cumreturn=False).fillna(method='ffill')
oct_data = bbg_to_df(bbg_data(otc_tickers, ['PX_LAST'], start_date, end_date), dict_aux_oct, cumreturn=False).fillna(method='ffill')


