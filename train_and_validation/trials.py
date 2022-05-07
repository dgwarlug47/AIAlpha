# docs complete
import tqdm
import time
import pprint
import os
import json
import optuna
import torch
from optuna.samplers import RandomSampler, TPESampler, NSGAIISampler
from tqdm import tqdm
from .validation.validation_rnn import new_rnn_train_test_validation_execution_collect_performance_metrics
from .helper_trial import avg_metric_getter, getter, save_optuna_plots, timeout, aggregate_feature_importance
from .train import get_trained_model
from .preparation_of_data.prepare import get_prepared_data, get_raw_data
from .helper_trial import hyper_parameter_check


def one_trial(data_sets, train_loader,
              model_hyper_parameters, info):
    # data sets are the train, test, validation data set
    # train_loader is the data loader used in the training
    # model_hyper_parameters is the hyper parameters of this specific model
    # this method trains the model and validates the model
    # the info is the general info the model
    hyper_parameter_check(model_hyper_parameters, info)
    model = get_trained_model(data_sets, train_loader,
                              model_hyper_parameters, info)

    model = model.eval()
    time7 = time.process_time()
    resu = new_rnn_train_test_validation_execution_collect_performance_metrics(
        model, data_sets,
        model_hyper_parameters, info)
    time8 = time.process_time()
    print("End of Validation process, total time: ", time8 - time7)
    return resu


def multiple_trials_optuna(search_space_creator, info):
    # num_trials is the amount of times you want to train and validate the models with the same hyper_parameters.
    # go to hyper_parameters.txt
    # cums is a list of the information you want to accumulate about of each trial.
    # In this application the cums are the performance_statistics, positions and returns.

    print('Getting raw data')
    raw_data = get_raw_data(info)
    torch.set_num_threads(info['num_threads'])
    info['current_experiment'] = -1
    not_constant_tracker = {}
    not_constant_analyzed_names = []
    def objective(trial):
        model_hyper_parameters = search_space_creator(trial)
        
        if len(not_constant_tracker) == 0:
            for analyzed_name in info['analyzed_names']:
                not_constant_tracker[analyzed_name] = model_hyper_parameters[analyzed_name]
        else:
            for analyzed_name in info['analyzed_names']:
                if model_hyper_parameters[analyzed_name] != not_constant_tracker[analyzed_name]:
                    if not(analyzed_name in not_constant_analyzed_names):
                        not_constant_analyzed_names.append(analyzed_name)

        for num_trial in range(info['num_trials']):
            info['current_experiment'] = info['current_experiment'] + 1
            if info['skipping_errors'] is True:
                try:
                    data_sets, train_loader = get_prepared_data(raw_data, model_hyper_parameters, info)
                except RuntimeError:
                    time.sleep(250)
                    continue
            else:
                data_sets, train_loader = get_prepared_data(raw_data, model_hyper_parameters, info)

            os.mkdir('./Experiments/' + info['prefix'] + '/' + str(info['current_experiment']))
            out_file = open('./Experiments/' + info['prefix'] + '/' + str(info['current_experiment']) + '/' + 'model_hyper_parameters', 'w')
            json.dump(model_hyper_parameters, out_file)
            out_file.close()

            print("STARTING trial " + str(num_trial))
            if info['skipping_errors'] is True:
                try:
                    one_trial(data_sets, train_loader, model_hyper_parameters, info=info)
                except:
                    print('there was an error with the following parameters')
                    print(model_hyper_parameters)
                    print(model_hyper_parameters)
                    continue
            else:
                one_trial(data_sets, train_loader, model_hyper_parameters, info=info)
        return avg_metric_getter(info, model_hyper_parameters)
    info['not_constant_analyzed_names'] = not_constant_analyzed_names

    if info['optuna_sampler'] == 'Random':
        sampler = RandomSampler()
    elif info['optuna_sampler'] == 'TPE':
        sampler = TPESampler()
    elif info['optuna_sampler'] == 'NSGAII':
        sampler = NSGAIISampler()
    else:
        raise "this optuna sampler is not currently provided"
    study = optuna.create_study(study_name = info['prefix'], direction="maximize", load_if_exists=True, sampler=sampler)
    study.optimize(objective, n_trials=info["num_shuffling"], n_jobs = info['optuna_n_jobs'])
    aggregate_feature_importance(info)
    save_optuna_plots(study, info)