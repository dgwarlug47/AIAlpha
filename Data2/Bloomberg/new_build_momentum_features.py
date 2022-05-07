import torch
import numpy as np
import pandas as pd
import csv
import time
import os
from momentum_features import simple_momentum, macd_momentum, chap_features, raw_returns_lags, prices_normalized, davi_features

new_directory = "All Futures Momentum Features Bloomberg14"

# this shoudld be the address of the file that is an CSV,
# and contains the prices of a bunch of assets.
# if I run the code: df = pd.read_csv(csv_data_file)
# then the first column of df should have the name 'date'
# and it should be all dates
# the rest of the columns should be the prices of the assets
# and the name of the asset should be the name of the column
csv_data_file = "all_futures.csv"

# S,L are for the macd
S = (8, 16, 32, 64)
L = (24, 48, 96, 192)

# for the macd
the_days = [21,3*21,6*21,12*21]

# self explanatory
max_lag_raw_returns = 63

# self explanatory
max_lag_prices_normalized = 100

# when you have zscores of prices you need to take them under a certain range
# for example price.rolling(20).mean()/price.rolling(20).std() this would be 20
rolling_window_prices_normalized = 35

dates_name = 'date'

playground = False

def downloadData(dict_):
  for ticker in (dict_.keys()):
    (dict_[ticker]).to_csv(ticker + '.csv')

def features_list_saver(fileName, features_list):
  with open(fileName, 'w') as f:
    for item in features_list:
      f.write("%s\n" % item)

all_data = pd.read_csv(csv_data_file)

all_data = all_data.drop('US10Y.1' , axis='columns')

size = all_data.shape[0]

dates = all_data[dates_name]

dict_ = {}

for i in range(size):
    dict_[i] = dates[i]

all_data = all_data.rename(index=dict_).drop([dates_name], axis=1)

prices = (all_data)

returns = prices.pct_change()

# prices and returns have dimensions (days, tickers)
# features is a dictionary where the key is the name of the feature,
# and the value is a dataframes of dimensions (days, num_tickers) of that feature

tickers = returns.columns

features_per_ticker = {}
normalized_returns_list = []
macd_list = []
chap_list = []
davi_list = []
raw_returns_lags_list = []
prices_normalized_list = []
first_time = True
for ticker in tickers:
    ticker_returns = returns[ticker]
    ticker_prices = prices[ticker]
    ticker_features = pd.DataFrame()
    ticker_features['returns'] = returns[ticker]
    
    for days in the_days:
        ticker_features['normalized_returns_' + str(days)] = (simple_momentum(ticker_returns, days))
        if first_time:
            normalized_returns_list.append('normalized_returns_' + str(days))

    for i in range(len(S)):
        ticker_features['MACD_short_' + str(S[i]) + '_long_' + str(L[i])] = (macd_momentum(ticker_prices, S[i], L[i]))
        if first_time:
            macd_list.append('MACD_short_' + str(S[i]) + '_long_' + str(L[i]))

    Price_df = pd.DataFrame()
    Price_df['Price'] = ticker_prices
    
    chap_df = chap_features(Price_df)
    ticker_features = pd.concat([ticker_features, chap_df], axis=1)
    if first_time:
      chap_list.extend(list(chap_df.columns))
    
    davi_df = davi_features(Price_df)
    ticker_features = pd.concat([ticker_features, davi_df], axis=1)
    if first_time:
      davi_list.extend(list(davi_df.columns))

    raw_returns_lags_df = raw_returns_lags(ticker_returns, max_lag_raw_returns)
    ticker_features = pd.concat([ticker_features, raw_returns_lags_df], axis=1)
    if first_time:
      raw_returns_lags_list.extend(list(raw_returns_lags_df.columns))

    prices_normalized_df = prices_normalized(ticker_prices, rolling_window_prices_normalized, max_lag_prices_normalized)
    ticker_features = pd.concat([ticker_features, prices_normalized_df], axis=1)
    if first_time:
      prices_normalized_list.extend(list(prices_normalized_df.columns))

    ticker_features.insert(0, 'returns next day', ticker_features['returns'].shift(-1).fillna(0))
    features_per_ticker[ticker] = ticker_features
    first_time = False

if not playground:
    features_list_saver('features_chap', chap_list)
    features_list_saver('features_macd', macd_list)
    features_list_saver('features_normalized_returns', normalized_returns_list)
    features_list_saver('features_raw_returns_lags', raw_returns_lags_list)
    features_list_saver('features_prices_normalized', prices_normalized_list)
    features_list_saver('features_davi', davi_list)

    # actually creating the raw label
    path = os.path.join(os.getcwd(), new_directory)

    try:
      os.rmdir(path)

    except:
      print('been trying to meet you')

    os.mkdir(path)

    os.chdir('./' + new_directory)

    downloadData(features_per_ticker)