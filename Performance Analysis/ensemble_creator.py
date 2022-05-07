import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import sys
import torch
import copy
from dill.source import getsource
from dill.temp import load_source
from deepdiff import DeepDiff
from ensemble import ensemble_position_getter, ensemble_weights_getter
from leaderboard import leaderboard
sys.path.append('./../train_and_validation/validation')
from helper_validation import save_performance_metrics_ensemble

# base_path is the list with all of the addresses of experiments you want to ensemble
base_paths = ['./../Experiments/glauber_input_normalization_no_importance2']
# the ensemble_path is the path where you want me to put your ensemble
ensemble_path = './Ensemble/glauber_new_Top_30_sanity_check'
# the method_type for the super learner, in this case we only have available
# simple_mean and simple_super_learner
method_type = 'simple_mean'
# if you put selection_type to be 0.3 it will make the ensemble under only the top 30% of
# the test.
selection_type = 0.3
# keep this as False
playground = False

if (isinstance(selection_type, str)) :
  assert(len(base_paths) == 1), 'the you select only one model, there must be only one base path'

def remove_trash(array):
    df = array
    df = df.fillna(0)
    df = df.replace(np.inf, 0)
    df = df.replace(-np.inf, 0)
    return df

def get_pnl_transaction_cost(df_pos, df_pnl, transaction_costs):
    num_days = df_pos.shape[0]
    arrays = [transaction_costs/10000 for _ in range(num_days)]
    repeated_transaction_cost = np.stack(arrays, axis=0)
    transaction_cost_pnl = (df_pnl - abs(repeated_transaction_cost*(df_pos - df_pos.shift(1))))
    transaction_cost_pnl = remove_trash(transaction_cost_pnl)
    return transaction_cost_pnl

def date_handler(pnl):
    pnl = pnl.rename(columns = {'Unnamed: 0': 'date'})
    new_days = list(pnl['date'])
    pnl = pnl.drop(columns= ['date'], axis=1)
    pnl.index = new_days
    return pnl

def loadTickers(fileName):
    # receives as input the address where the names of the tickers are stored.
    # outputs the list of the tickers names.
    lineList = list()
    with open(fileName) as f:
        for line in f:
            lineList.append(line)
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList

def remove_trash(my_array):
    my_array[my_array == -np.inf] = 0
    my_array[my_array == np.inf] = 0
    my_array[my_array != my_array] = 0
    return my_array

try:
  os.rmdir(ensemble_path)

except:
  print('been trying to meet you')

if not playground:
  os.mkdir(ensemble_path)

# trial_summary_dict has the keys as the experiments names, and as the values its trial summaries
trial_summary_dict = {}

leaderboard_df = leaderboard(base_paths)

if not playground:
  leaderboard_df.to_csv(ensemble_path + '/leaderboard.csv')

leaderboard_df = leaderboard_df.sort_values(by = 'sharpe_of_average_transaction_cost_test' , ascending=False)

number_of_guys = len(leaderboard_df.index)

if not(isinstance(selection_type, str)):
  number_of_selected_guys = int(number_of_guys * selection_type)
  selected_paths = list((leaderboard_df.iloc[0:number_of_selected_guys, :]).index)

else:
  selected_paths = [base_paths[0] + '/' + selection_type]

info = None
for path in selected_paths:
  text_file = open(path + '/../info', "r")

  #read whole file to a string

  info_str = text_file.read()

  exec(info_str)
  if info is None:
    info = info_creator()
  #else:
    #if (DeepDiff(info, info_creator())):
    #  raise "all infos should be the same"

  text_file.close()

name_file_path = './../' + info["tickers_names_file_path"]
Names_list = loadTickers(name_file_path)
base_path = './../' + info["tickers_base_path"]
next_day_real_returns = pd.DataFrame()
days = None
for name in Names_list:
    path_for_real_returns = base_path + name + '.csv'
    Data = pd.read_csv(path_for_real_returns)
    Data = Data.rename(columns = {'Unnamed: 0': 'date'})
    # Data should have dimensions (num_data_points, features + 1)
    # Data[:, 0] is the real_returns, where Data[:, 1:] is the features
    new_days = list(Data['date'])
    if days is None:
        days = new_days
    else:
        if(days != new_days):
            print(name)
            raise('all tickers should be from the same time')
    Data = Data.drop(columns= ['date'], axis=1)
    assert(Data.columns[0] == 'returns next day'), 'the first column needs to be the returns of the next day'
    next_day_real_returns[name] = Data.iloc[:, 0]
next_day_real_returns.index = days

all_returns = next_day_real_returns

train_pos_list = []
test_pos_list = []
validation_pos_list = []
for complete_path in selected_paths:
  if 'padres' in complete_path:
    train_pos_list.append(date_handler(pd.read_csv(complete_path + '/pos of next day train.csv' )))
    test_pos_list.append(date_handler(pd.read_csv(complete_path + '/pos of next day test.csv')))
    validation_pos_list.append(date_handler(pd.read_csv(complete_path + '/pos of next day validation.csv')))
  else:
    train_pos_list.append(date_handler(pd.read_csv(complete_path + '/position of next day/train.csv' )))
    test_pos_list.append(date_handler(pd.read_csv(complete_path + '/position of next day/test.csv')))
    validation_pos_list.append(date_handler(pd.read_csv(complete_path + '/position of next day/validation.csv'))) 

  tickers = validation_pos_list[-1].columns
  
  train_begin = min(train_pos_list[-1].index)
  train_end = max(train_pos_list[-1].index)

  train_returns = all_returns[all_returns.index >= train_begin]
  train_returns = train_returns[train_returns.index <= train_end]
  train_returns = train_returns[tickers]
  train_returns = train_returns.fillna(0)

  test_begin = min(test_pos_list[-1].index)
  test_end = max(test_pos_list[-1].index)

  test_returns = all_returns[all_returns.index >= test_begin]
  test_returns = test_returns[test_returns.index <= test_end]
  test_returns = test_returns[tickers]
  test_returns = test_returns.fillna(0)

  validation_begin = min(validation_pos_list[-1].index)
  validation_end = max(validation_pos_list[-1].index)

  validation_returns = all_returns[all_returns.index >= validation_begin]
  validation_returns = validation_returns[validation_returns.index <= validation_end]
  validation_returns = validation_returns[tickers]
  validation_returns = validation_returns.fillna(0)

x_final = ensemble_weights_getter(test_pos_list, test_returns, method_type, info)

train_ensemble_pos = ensemble_position_getter(train_pos_list, x_final)
train_no_transaction_cost_pnl = train_ensemble_pos * train_returns
np_train_ensemble_pos = np.array(train_ensemble_pos.T)
np_train_no_transaction_cost_pnl = np.array(train_no_transaction_cost_pnl.T)
info['train_days'] = list(train_ensemble_pos.index)

test_ensemble_pos = ensemble_position_getter(test_pos_list, x_final)
test_no_transaction_cost_pnl = test_ensemble_pos * test_returns
np_test_ensemble_pos = np.array(test_ensemble_pos.T)
np_test_no_transaction_cost_pnl = np.array(test_no_transaction_cost_pnl.T)
info['test_days'] = list(test_ensemble_pos.index)

validation_ensemble_pos = ensemble_position_getter(validation_pos_list, x_final)
validation_no_transaction_cost_pnl = validation_ensemble_pos * validation_returns
np_validation_ensemble_pos = np.array(validation_ensemble_pos.T)
np_validation_no_transaction_cost_pnl = np.array(validation_no_transaction_cost_pnl.T)
info['validation_days'] = list(validation_ensemble_pos.index)

info['tickers_names_file_path'] = './../' + info['tickers_names_file_path']
if not playground:
  save_performance_metrics_ensemble([
  [np_train_ensemble_pos, np_test_ensemble_pos, np_validation_ensemble_pos],
  [np_train_no_transaction_cost_pnl, np_test_no_transaction_cost_pnl, np_validation_no_transaction_cost_pnl],
  [np_train_ensemble_pos, np_test_ensemble_pos, np_validation_ensemble_pos]], ensemble_path, info)
  validation_no_transaction_cost_pnl.to_csv(ensemble_path + '/no transaction cost pnl.csv')