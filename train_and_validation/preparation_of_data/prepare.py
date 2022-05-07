import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import pandas as pd
from random import randrange
from .labelling import labelling, leverage
from .helper_preparation_of_data import loadTickers, AssetsData, store_and_show_true_distribution, save_mean_and_variance, new_train_test_validation_data_creator, remove_trash


def get_raw_data(info):
    # Names_list should be the names of the tickers that you are analyzing
    # data_preparation_hyper_parameters is the usual, the base_path
    # plus the ticker name should give the path to acess the data.
    # the first element it returns has shape (number of tickers,
    # length of tickers, num_features)
    #  the second element it returns has shape (number of tickers,
    # lenght of tickers), the second is the returns and the first
    # is the features and the label they aim to predict
    # in the case the model is end to end, the label is simply
    # the returns, and in the probability mode it is the discrete
    # label of its respective return
    name_file_path = info["tickers_names_file_path"]
    Names_list = loadTickers(name_file_path)
    info['num_tickers'] = len(Names_list)
    base_path = info["tickers_base_path"]
    features_names_file_paths = info['features_names_file_paths']
    features_list = []
    for features_names_file_path in features_names_file_paths:
        new_features_list = loadTickers(features_names_file_path)
        if bool(set(new_features_list) & set(features_list)):
            raise "the features should have no intersection"
        features_list.extend(new_features_list)
    assert(not ('returns next day' in features_list)), 'this is clearly forward looking bias'
    info['features_names'] = features_list
    Data_list = []
    next_day_real_returns_list = []
    days = None
    for name in Names_list:
        path_for_real_returns = base_path + name + '.csv'
        Data = pd.read_csv(path_for_real_returns)
        Data = Data.rename(columns = {'Unnamed: 0': 'date'})
        Data = Data.query("date >="+ "'" + info['start_date'] + "'")
        Data = Data.query("date <"+ "'" + info['end_date'] + "'")
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
        to_keep = ['returns next day'] + features_list
        Data = Data[to_keep]
        assert(Data.columns[0] == 'returns next day'), 'the first column needs to be the returns of the next day'
        if info['no_last_day']:
            Data[features_list] = Data[features_list].shift(1)
            Data = Data.fillna(0)
        info['num_features'] = Data.shape[1] - 1
        next_day_real_returns = np.array(copy.deepcopy(Data.iloc[:, 0]))
        Data.insert(1, 'leverage_factor', leverage(next_day_real_returns, info))
        Data = np.array(Data)
        Data_list.append(Data)
        next_day_real_returns_list.append(next_day_real_returns)
    info['all_days'] = new_days
    all_next_day_real_returns = np.stack(next_day_real_returns_list, axis=0)
    all_x = np.stack(Data_list, axis=0)
    all_x = remove_trash(all_x)
    all_next_day_real_returns = remove_trash(all_next_day_real_returns)
    return [all_x, all_next_day_real_returns, days]


def get_prepared_data(raw_data, data_preparation_hyper_parameters, info):
    # raw_data is a list with 2 elements, the first has dimensions (num_tickers, num_days, num_features + 2)
    # the second is the real_returns has dimensions (num_tickers, num_days)
    # data_preparation_hyper_parameters is simply a way to customize the preparation of data
    # this function returns in order: [train_data, test_data], real_returns
    # Note that the train_data and the test_data have the shape: num_of_securities, num_data_points/seq_len, seq_len, num_features
    # real_returns in the other hand have the shape: num_securities, num_data_points
    # the real_returns comes from the validation_address not the train address
    # the first feature of train_data and test_data is the discretized returns, whereas the real returns only has one feature, which is the real returns
    time0 = time.process_time()

    if not(info['end_date'] > info['validation_begin'] and info['validation_begin'] > info['test_begin'] and info['test_begin'] > info['start_date']):
        raise "validation begining before test is forward looking bias"

    """ part 1: retreving the information"""
    device = info['device']

    """ part 2: getting the tickers names from both training and validation addresses """

    """ part 3: getting the data from each ticker from both the training and validation addresses """
    all_x = raw_data[0]
    all_x[:,  :, 0] = labelling(all_x[:, :, 0], data_preparation_hyper_parameters['label_type'])
    next_day_real_returns = raw_data[1]
    train_data, test_data, validation_data, next_day_train_returns, next_day_test_returns, next_day_validation_returns = new_train_test_validation_data_creator(all_x, next_day_real_returns, data_preparation_hyper_parameters, info)
    copy_train_data = np.copy(train_data)
    copy_test_data = np.copy(test_data)
    copy_validation_data = np.copy(validation_data)
    # train_data, test_data, validation_data, have dimensions: (num_securities, num_of_days, num_of_features + 2)
    # the following two lines of code do not remove the artifcial data.

    """ part 4: bagging the training set """
    assetsTrainData = AssetsData(train_data, data_preparation_hyper_parameters['seq_length'], data_preparation_hyper_parameters['bootstrap_ratio'], data_preparation_hyper_parameters['bootstrap_seed'], device)

    """ part 5: saving the mean and the variance of the data set features """
    save_mean_and_variance(train_data, info)

    time1 = time.process_time()
    print("End of the preparation data, ", " this was the total time:", time1 - time0)
    return [ [torch.FloatTensor(copy_train_data).to(device), next_day_train_returns], [torch.FloatTensor(copy_test_data).to(device), next_day_test_returns], [torch.FloatTensor(copy_validation_data).to(device) , next_day_validation_returns]], assetsTrainData
