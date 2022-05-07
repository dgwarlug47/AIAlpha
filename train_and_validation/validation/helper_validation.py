# doc complete.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import empyrical as ep
import seaborn as sns
import os
import json
import warnings


def loadTickers(fileName):
    # receives as input the address where the names of the tickers are stored.
    # outputs the list of the tickers names.
    lineList = list()
    with open(fileName) as f:
        for line in f:
            lineList.append(line)
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList


def pos_statistics_extractor(pos_list):
    """ part 3 extracting average and variance of the positions"""
    expected_value = np.array([[0., 0.], [0., 0.], [0., 0.]])
    variance = np.array([[0., 0.], [0., 0.], [0., 0.]])
    all_prob = np.array([[0., 0.], [0., 0.], [0., 0.]])

    for train_or_test_or_validation in range(3):
        for bottom_or_top in range(2):
            if bottom_or_top == 0:
                filtered_positions = (pos_list[train_or_test_or_validation] < 0)
            else:
                filtered_positions = (pos_list[train_or_test_or_validation] >= 0)
            all_prob[train_or_test_or_validation][bottom_or_top] = np.nanmean(filtered_positions)
            expected_value[train_or_test_or_validation][bottom_or_top] = (pos_list[train_or_test_or_validation])[filtered_positions].mean()
            variance[train_or_test_or_validation][bottom_or_top] = (pos_list[train_or_test_or_validation])[filtered_positions].std()

    all_probs_bottom = all_prob[:, 0]
    all_probs_top = all_prob[:, 1]
    expected_value_bottom = expected_value[:, 0]
    expected_value_top = expected_value[:, 1]
    variance_bottom = variance[:, 0]
    variance_top = variance[:, 1]

    pos_statistics = {"alpha_probs_bottom": all_probs_bottom, "alpha_probs_top": all_probs_top,
            "expected_value_bottom": expected_value_bottom, "expected_value_top": expected_value_top,
            "variance_top": variance_top, "variance_bottom": variance_bottom}
    return pos_statistics


def pnl_statistics_extractor(pnl_list):
    # validaiton_num_of_securities - self explanatory.
    # pos_list is the list of positions that were taken, it should have dimensions like (validation_num_securities, num_data_points).
    # pnl_list the returns each security and each day positions, dimensions: (validation_num_securities, num_data_points).
    num_of_securities = pnl_list[0].shape[0]

    """ extracting the pnl metrics """
    """ sharpe per security """
    Sharp_per_security = [0., 0., 0.]
    Sharp_per_security[0] = np.sqrt(252)*(np.mean(pnl_list[0], axis=1) / np.std(pnl_list[0], axis=1))
    Sharp_per_security[1] = np.sqrt(252)*(np.mean(pnl_list[1], axis=1) / np.std(pnl_list[1], axis=1))
    Sharp_per_security[2] = np.sqrt(252)*(np.mean(pnl_list[2], axis=1) / np.std(pnl_list[2], axis=1))

    average_sharpe_train = np.nanmean(Sharp_per_security[0])
    average_sharpe_test = np.nanmean(Sharp_per_security[1])
    average_sharpe_validation = np.nanmean(Sharp_per_security[2])
    average_sharp = np.array([average_sharpe_train, average_sharpe_test, average_sharpe_validation])

    # portifolio as the average
    train_portifolio = (pnl_list[0].sum(axis=0) / num_of_securities)
    test_portifolio = (pnl_list[1].sum(axis=0) / num_of_securities)
    validation_portifolio = (pnl_list[2].sum(axis=0) / num_of_securities)

    # sharpe ratio
    sharpe_of_average_train = np.sqrt(252)*train_portifolio.mean() / train_portifolio.std()
    sharpe_of_average_test = np.sqrt(252)*test_portifolio.mean() / test_portifolio.std()
    sharpe_of_average_validation = np.sqrt(252)*validation_portifolio.mean() / validation_portifolio.std()
    sharpe_of_average = np.array([sharpe_of_average_train, sharpe_of_average_test, sharpe_of_average_validation])

    # sortino ratio
    sortino_of_average_train = ep.stats.sortino_ratio(train_portifolio, period='daily')
    sortino_of_average_test = ep.stats.sortino_ratio(test_portifolio, period='daily')
    sortino_of_average_validation = ep.stats.sortino_ratio(validation_portifolio, period='daily')
    sortino_of_average = np.array([sortino_of_average_train, sortino_of_average_test, sortino_of_average_validation])

    # calmar ratio
    calmar_of_average_train = ep.stats.sortino_ratio(train_portifolio, period='daily')
    calmar_of_average_test = ep.stats.sortino_ratio(test_portifolio, period='daily')
    calmar_of_average_validation = ep.stats.sortino_ratio(validation_portifolio, period='daily')
    calmar_of_average = np.array([calmar_of_average_train, calmar_of_average_test, calmar_of_average_validation])

    # percentage of positive returns
    percentage_of_positive_returns_train = len(train_portifolio[train_portifolio > 0])/(len(train_portifolio))
    percentage_of_positive_returns_test = len(test_portifolio[test_portifolio > 0])/(len(test_portifolio))
    percentage_of_positive_returns_validation = len(validation_portifolio[validation_portifolio > 0])/(len(validation_portifolio))
    percentage_of_positive_returns = np.array([percentage_of_positive_returns_train, percentage_of_positive_returns_test, percentage_of_positive_returns_validation])
    
    # average returns of each ticker
    rets_of_average_train = np.sum(train_portifolio)
    rets_of_average_test = np.sum(test_portifolio)
    rets_of_average_validation = np.sum(validation_portifolio)
    rets_of_average = np.array([rets_of_average_train, rets_of_average_test, rets_of_average_validation])
    
    # average returns of the portifolio as the average
    average_rets_train = np.mean(pnl_list[0])
    average_rets_test = np.mean(pnl_list[1])
    average_rets_validation = np.mean(pnl_list[2])
    average_rets = np.array([average_rets_train, average_rets_test, average_rets_validation])

    """ part 5 wrap up and printing the trial summary """
    pnl_statistics = {"sharpe_of_average": sharpe_of_average, "rets_of_average": rets_of_average,
            "average_sharp": average_sharp, "average_rets": average_rets,
            "sortino_of_average": sortino_of_average, "calmar_of_average": calmar_of_average, 
            "percentage_of_positive_returns": percentage_of_positive_returns}
    return pnl_statistics


def remove_trash(array):
    assert(type(array) == np.ndarray)
    df = pd.DataFrame(array).T
    df = df.fillna(0)
    df = df.replace(np.inf, 0)
    df = df.replace(-np.inf, 0)
    return np.array(df.T)


def get_pos_pnl_list_volatility_scaled(pos_list, pnl_list, vol_tgt):
    new_pos_list = []
    new_pnl_list = []
    for index in range(len(pos_list)):
        new_pos, new_pnl = get_pos_pnl_volatility_scaled(pos_list[index], pnl_list[index], vol_tgt)
        new_pos_list.append(new_pos)
        new_pnl_list.append(new_pnl)
    return new_pos_list, new_pnl_list


def get_pos_pnl_volatility_scaled(pos, pnl, vol_tgt):
    original_returns = pd.DataFrame(pnl/pos).T
    vol_scale = np.array((original_returns.ewm(span=60).std()/vol_tgt).T)
    with warnings.catch_warnings():
    # this will suppress all warnings in this block
        warnings.simplefilter("ignore")
        new_pos = (pos/vol_scale)
    new_pos = remove_trash(new_pos)
    new_pnl = (new_pos*np.array(original_returns.T))
    new_pnl = remove_trash(new_pnl)
    return new_pos, new_pnl


def get_pnl_list_transaction_cost(pos_list, pnl_list, transaction_costs):
    transaction_cost_pnl_list = []
    for index in range(len(pos_list)):
        transaction_cost_pnl_list.append(get_pnl_transaction_cost(pos_list[index], pnl_list[index], transaction_costs))
    return transaction_cost_pnl_list


def get_pnl_transaction_cost(pos, pnl, transaction_costs):
    df = pd.DataFrame(pos).T
    transaction_cost_pnl = (pnl - np.abs(np.expand_dims(transaction_costs, axis=1)*(np.array(    (df - df.shift(1)).T  ))))
    transaction_cost_pnl = remove_trash(transaction_cost_pnl)
    return transaction_cost_pnl


def trial_summary_extractor(pos_list, pnl_list, unnormalized_pos_list, info):
    trial_summary = {}
    trial_summary.update(pos_statistics_extractor(pos_list))
    trial_summary.update(pnl_statistics_extractor(pnl_list))
    transaction_cost_pnl_list = get_pnl_list_transaction_cost(pos_list, pnl_list, info['test_transaction_cost']/10000)
    transaction_cost_trial_summary = pnl_statistics_extractor(transaction_cost_pnl_list)
    for key in transaction_cost_trial_summary.keys():
        trial_summary[key + '_transaction_cost'] = transaction_cost_trial_summary[key]
    unnormalized_pos_summary = pos_statistics_extractor(unnormalized_pos_list)
    for key in unnormalized_pos_summary:
        trial_summary[key + '_unnormalized'] = unnormalized_pos_summary[key]
    return trial_summary


def save_performance_metrics(lists, info):
    pos_list = lists[0]
    pnl_list = lists[1]
    unnormalized_pos_list = lists[2]
    importances = lists[3]
    importances_dict = lists[4]
    trial_summary = trial_summary_extractor(pos_list, pnl_list, unnormalized_pos_list, info)

    current_experiment = info['current_experiment']
    prefix = info['prefix']
    path = 'Experiments/' + prefix + '/' + str(current_experiment) + '/'
    if not isinstance(importances, int):
        save_importance_data(importances, importances_dict, path, info)
    save_json(serializable_transformation(trial_summary), path + 'trial_summary.txt')
    save_poses_and_pnl(pos_list[0], unnormalized_pos_list[0], pnl_list[0], 'train', path, info)
    save_poses_and_pnl(pos_list[1], unnormalized_pos_list[1], pnl_list[1], 'test', path, info)
    save_poses_and_pnl(pos_list[2], unnormalized_pos_list[2], pnl_list[2], 'validation', path, info)

    return trial_summary


def save_performance_metrics_ensemble(lists, ensemble_path, info):
    pos_list = lists[0]
    pnl_list = lists[1]
    unnormalized_pos_list = lists[2]
    trial_summary = trial_summary_extractor(pos_list, pnl_list, unnormalized_pos_list, info)

    path = ensemble_path + '/'
    save_json(serializable_transformation(trial_summary), path + 'trial_summary.txt')
    save_poses_and_pnl(pos_list[0], unnormalized_pos_list[0], pnl_list[0], 'train', path, info)
    save_poses_and_pnl(pos_list[1], unnormalized_pos_list[1], pnl_list[1], 'test', path, info)
    save_poses_and_pnl(pos_list[2], unnormalized_pos_list[2], pnl_list[2], 'validation', path, info)

    return trial_summary


def save_importance_data(importances, importances_dict, path, info):
    features_importances = importances[0]
    lags_importances = importances[1]
    tickers_importances = importances[2]
    os.mkdir('./' + path + '/model explainability')
    features_importances_dict = {}
    features_importances_df = pd.DataFrame()
    for index in range(len(features_importances)):
        features_importances_dict[info['features_names'][index]] = features_importances[index]
        features_importances_df[info['features_names'][index]] = [features_importances[index]]
    features_importances_df = features_importances_df.T
    features_importances_df[1] = features_importances_df.index
    lags_importances_dict = {}
    save_plot_df(pd.DataFrame(lags_importances), 'lags importances', './' + path + '/model explainability/lags_importances', 'lags importances')
    save_plot_df(pd.DataFrame(lags_importances).cumsum(), 'cumsum lags importances', './' + path + '/model explainability/cumsum_lags_importances', 'cumsum lags importances')
    save_json(importances_dict, './' + path + '/model explainability/importances_dict.txt')
    save_json(features_importances_dict, './' + path + '/model explainability/feature_importances_dict.txt')
    save_bar_plot(features_importances_df, x=0, y=1, fname='./' + path + '/model explainability/feature_importance_graph')


def save_poses_and_pnl(pos, unnormalized_pos, pnl, train_test_or_validation, path, info):
    if train_test_or_validation == "test":
        days = info['test_days']
    elif train_test_or_validation == "validation":
        days = info['validation_days']
    elif train_test_or_validation == "train":
        days = info['train_days']
    tickers = loadTickers(info['tickers_names_file_path'])
    new_columns_dict = {}
    new_index_dict = {}

    for i in range(pnl.shape[1]):
        new_index_dict[i] = days[i]

    for i in range(len(tickers)):
        new_columns_dict[i] = tickers[i]
    
    place_holder1 = (pd.DataFrame(pnl)).T
    place_holder1 = place_holder1.rename(columns=new_columns_dict)
    place_holder1 = place_holder1.rename(index=new_index_dict)
    
    place_holder2 = (pd.DataFrame(pos)).T
    place_holder2 = place_holder2.rename(columns=new_columns_dict)
    place_holder2 = place_holder2.rename(index=new_index_dict)

    place_holder3 = (pd.DataFrame(unnormalized_pos)).T
    place_holder3 = place_holder3.rename(columns=new_columns_dict)
    place_holder3 = place_holder3.rename(index=new_index_dict)
    
    directory = './' + path + 'pnl cumsum per ticker'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = './' + path + 'pnl cumsum average'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = './' + path + 'position of next day'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = './' + path + 'before leverage position of next day'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = './' + path + 'before leverage position per ticker graph'
    if not os.path.exists(directory):
        os.mkdir(directory)
 
    save_plot_df(place_holder1.cumsum(), 'pnl cumsum per ticker ' + train_test_or_validation, './' + path + 'pnl cumsum per ticker/' + train_test_or_validation, 'cumsum')
    save_plot_df(place_holder1.mean(axis=1).cumsum(), 'pnl cumsum average ' + train_test_or_validation, './' + path + 'pnl cumsum average/' + train_test_or_validation, 'cumsum')
    place_holder2.to_csv('./' + path + 'position of next day/' + train_test_or_validation + '.csv')
    place_holder3.to_csv('./' + path + 'before leverage position of next day/ ' + train_test_or_validation + '.csv')
    save_plot_df(place_holder3, 'before leverage position per ticker ' + train_test_or_validation, './' + path + 'before leverage position per ticker graph/' + train_test_or_validation, 'position')


def save_bar_plot(data, x, y, fname):
    #sns.set_context('paper')
    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    sns.barplot(x = x, y = y, data = data)
    plt.savefig(fname)


def save_plot_df(df, title, fname, y):
    plt.figure()
    #f.set_figwidth(20) 
    #f.set_figheight(20)
    df.plot()
    plt.title(title)
    plt.ylabel(y)
    plt.savefig(fname)


def sample_returns(sample_np, bins, using_middle_returns):
    # sample_np is a numpy array of labels.
    # bins is an array where, bins[i] is the return of the
    # ith label.
    rets = []
    for i in range(len(sample_np)):
        if sample_np[i] == 0:
            ret = bins[0]
        elif sample_np[i] == len(bins):
            ret = bins[len(bins) - 1]
        else:
            lim_inf = bins[sample_np[i] - 1]
            lim_sup = bins[sample_np[i]]
            if using_middle_returns:
                std = lim_inf + (lim_sup - lim_inf) * 0.5
            else:
                std = lim_inf + (lim_sup - lim_inf) * np.random.random()
            ret = std
        rets.append(ret)
    return np.array(rets)


def find_alpha2(prob, rets, alphas):
    # in this instance, prob has three dimensions, (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
    # rets should have  the two dimensions (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
    U_list = []
    for alpha in alphas:
        U = prob * np.log(1.0 + alpha * rets)
        U_list.append(np.sum(U, axis=1))
    U_vec = np.array(U_list)
    # print("----------", U_vec.shape)
    return alphas[np.argmax(U_vec, axis=0)]


def comparison(pos1, pos2, likelihoods, rets):
    # likelihoods has the 2 dimensions, (batch_size, num_of_distribution_bins)
    # rets has the same dimensions.
    z = 0
    bad_times = 0
    for likelihood, ret, po1, po2 in zip(likelihoods, rets, pos1, pos2):
        z = z + 1
        loss1 = - np.dot(likelihood, np.log(1 + ret * po1))
        loss2 = - np.dot(likelihood, np.log(1 + ret * po2))
        if (loss1 > loss2):
            bad_times = bad_times + 1
    print('be aware of the bad times', bad_times)
    input("see bad times")
    # print("this is the optimal alpha", alpha)
    return


def save_json(obj, name):
    with open(name, "w") as f:
        json.dump(obj, f, skipkeys=True, indent=True)


def serializable_transformation(obj):
    for key in obj.keys():
        obj[key] = list(obj[key])
    return obj