import os
import ast
import pandas as pd

def leaderboard(base_paths):
    # trial_summary_dict has the keys as the experiments names, and as the values its trial summaries
    trial_summary_dict = {}
    model_hyper_parameters_dict = {}

    for base_path in base_paths:
        for filename in os.listdir(base_path):
            if not filename.isnumeric():
                continue 
            complete_path = base_path + '/' + filename
            if not (os.path.isdir(complete_path)):
                continue

            text_file = open(complete_path + '/trial_summary.txt' , "r")
            # read whole file to a string
            trial_summary_str = text_file.read()
            clean_trial_summary_str = trial_summary_str.replace('NaN', '0')
            text_file.close()
            trial_summary = ast.literal_eval(clean_trial_summary_str)
            trial_summary_dict[complete_path] = trial_summary

            text_file2 = open(complete_path + '/model_hyper_parameters')
            #read whole file to a string
            model_hyper_parameters_str = text_file2.read()
            clean_model_hyper_parameters_str = model_hyper_parameters_str.replace('NaN', '0')
            clean_model_hyper_parameters_str = clean_model_hyper_parameters_str.replace('true', 'True')
            clean_model_hyper_parameters_str = clean_model_hyper_parameters_str.replace('false', 'False')
            clean_model_hyper_parameters_str = clean_model_hyper_parameters_str.replace('null', 'None')
            text_file2.close()
            model_hyper_parameters = ast.literal_eval(clean_model_hyper_parameters_str)
            model_hyper_parameters_dict[complete_path] = model_hyper_parameters

    for z in trial_summary_dict.keys():
        metrics = trial_summary_dict[z].keys()
        metrics = list(metrics)
        break
    for z in model_hyper_parameters_dict.keys():
        hyper_parameters = model_hyper_parameters_dict[z].keys()
        hyper_parameters = list(hyper_parameters)
    indexes = list(trial_summary_dict.keys())

    # the columns should be the metric and, the rows the experiments
    leaderboard_df = pd.DataFrame()

    for metric in metrics:
        for train_validation_test in ['train', 'test', 'validation']:
            new_metric = metric + '_' + train_validation_test
            all_guys = []
            for experiment_name in indexes:
                experiment_dict = trial_summary_dict[experiment_name]
                if train_validation_test == 'train':
                    all_guys.append(experiment_dict[metric][0])
                elif train_validation_test == 'test':
                    all_guys.append(experiment_dict[metric][1])
                elif train_validation_test == 'validation':
                    all_guys.append(experiment_dict[metric][2]) 
            leaderboard_df[new_metric] = all_guys
    for hyper_parameter in hyper_parameters:
        all_guys = []
        for experiment_name in indexes:
            experiment_dict = model_hyper_parameters_dict[experiment_name]
            all_guys.append(experiment_dict[hyper_parameter])
        leaderboard_df[hyper_parameter] = all_guys

    #first_col = leaderboard.pop('sharpe_of_average_vol_scale_transaction_cost_validation')
    #second_col = leaderboard.pop('sharpe_of_average_vol_scale_transaction_cost_test')
    #third_col = leaderboard.pop('sharpe_of_average_vol_scale_transaction_cost_train')

    #leaderboard.insert(0, 'sharpe_of_average_vol_scale_transaction_cost_train', third_col)
    #leaderboard.insert(0, 'sharpe_of_average_vol_scale_transaction_cost_test', second_col)
    #leaderboard.insert(0, 'sharpe_of_average_vol_scale_transaction_cost_validation', first_col)

    leaderboard_df.index = indexes

    #leaderboard = leaderboard.sort_values(by = 'sharpe_of_average_vol_scale_transaction_cost_test', ascending=False)

    return leaderboard_df