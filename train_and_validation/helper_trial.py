import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import optuna
import ast
from random import randrange
from functools import wraps
import errno
import os
import signal
import seaborn as sns

def save_optuna_plots(study, info):
    os.mkdir("./Experiments/" + info['prefix'] + '/optuna_graphs')
    for x in info['analyzed_names']:
        for y in info['analyzed_names']:
            if x >= y:
                continue
            fig = optuna.visualization.plot_contour(study, params=[x, y])
            fig.write_image("./Experiments/" + info['prefix'] + '/optuna_graphs/plot_countour_' + x + '_' + y + '.png')
    fig = optuna.visualization.plot_param_importances(study, params = info['not_constant_analyzed_names'])
    fig.write_image("./Experiments/" + info['prefix'] + '/optuna_graphs/hyper_parameter_importance' + '.png')
    if len(info['not_constant_analyzed_names']) != 0:
        fig = optuna.visualization.plot_slice(study, info['not_constant_analyzed_names'])
        fig.write_image("./Experiments/" + info['prefix'] + '/optuna_graphs/slice.png')


def avg_metric_getter(info, model_hyper_parameters):
    optuna_metric_name = info['optuna_metric_name']
    metric_sum = 0
    counter = 0
    for num_trial in range(info['num_trials']):
        try:
            with open('./Experiments/' + info['prefix'] + '/' + str(info['current_experiment']) + '/trial_summary.txt') as f:
                new_metric = json.load(f)[optuna_metric_name][1]
            metric_sum = metric_sum + new_metric
            counter = counter + 1
        except:
            print('something went wrong')
    try:
        return metric_sum/counter
    except:
        return 0


def dict_sum(a,b):
    if len(a) != 0 and len(b) != 0:
        assert(list(a.keys()) == list(b.keys())), "weird stuff going on"
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    c = {}
    for key in b.keys():
        c[key] = a[key] + b[key]
    return c


def aggregate_feature_importance(info):
    minutes = info['feature_importance_minutes']
    if minutes == 0:
        return
    base_path = './Experiments/' + info['prefix']
    expl_path = base_path + '/model explainability'
    os.mkdir(expl_path)
    sum_features_importances_dict = {}
    for filename in os.listdir(base_path):
        if not (filename.isnumeric()):
            continue
        complete_path = base_path + '/' + filename + '/model explainability'
        if not (os.path.isdir(complete_path)):
            continue
        text_file = open(complete_path + '/feature_importances_dict.txt' , "r")
        # read whole file to a string
        features_importances_dict = text_file.read()
        clean_features_importances_dict = features_importances_dict.replace('NaN', '0')
        text_file.close()
        features_importances_dict = ast.literal_eval(clean_features_importances_dict)
        sum_features_importances_dict = dict_sum(sum_features_importances_dict, features_importances_dict)
    features_importances_df = pd.DataFrame()
    for feature_name in info['features_names']:
        features_importances_df[feature_name] = [sum_features_importances_dict[feature_name]]
    features_importances_df = features_importances_df.T
    features_importances_df[1] = features_importances_df.index
    save_bar_plot(features_importances_df, 0, 1, './' + expl_path + '/feature_importance_graph')


def hyper_parameter_check(hyper_parameters, info):
    #assert(info['feature_importance_minutes'] < 1e-5 or hyper_parameters['input_normalization']), "input normalization has to be true when calculating feature importance"
    assert((not info['long_only_benchmark'] and hyper_parameters['epochs'] > 0) or (info['long_only_benchmark'] and hyper_parameters['epochs']==0)), "becareful with long only benchmark"


def save_bar_plot(data, x, y, fname):
    #sns.set_context('paper')
    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    sns.barplot(x = x, y = y, data = data)
    plt.savefig(fname)


def save_json(obj, name):
    with open(name, "w") as f:
        json.dump(obj, f, skipkeys=True, indent=True)


def list_iterator(lists):
    if len(lists) == 1:
        return map(lambda x: [x], lists[0])
    else:
        all_things = []
        things = list_iterator(lists[1:])
        for thing in things:
            for new_guy in lists[0]:
                all_things.append([new_guy] + thing)
        return all_things


def grid_hyper_parameters_iterator(lists):
    if len(lists.keys()) == 1:
        name = list(lists.keys())[0]
        return [{name: ll} for ll in lists[name]]
    else:
        all_things = []
        all_but_first = lists.copy()
        first_name = list(lists.keys())[0]
        del all_but_first[first_name]
        things = grid_hyper_parameters_iterator(all_but_first)
        for thing in things:
            for new_guy in lists[first_name]:
                new_thing = thing.copy()
                new_thing[first_name] = new_guy
                all_things.append(new_thing)
        return all_things


def random_hyper_parameters_iterator(lists, num_shuffles):
    import time
    all_things = []
    counter = 0
    time1 = time.process_time()
    while True:
        if (counter == num_shuffles):
            break
        thing = {}
        for key in lists:
            size = len(lists[key])
            index = randrange(size)
            thing[key] = lists[key][index]
        already = False
        for previous_thing in all_things:
            if (previous_thing == thing):
                already = True
                continue
        if not already:
            all_things.append(thing)
        counter = counter + 1
        time2 = time.process_time()
        if (time2 - time1 > 60):
          raise "You put more shuffles than possible hyper parameters"
    return all_things


def searcher(lists, title):
    num_shuffles_list = lists[title]
    if (len(num_shuffles_list) != 1):
        raise "You can only select the parameter hyper_parameter_num_shuffles once"
    num_shuffles = num_shuffles_list[0]
    if num_shuffles != 0:
        return random_hyper_parameters_iterator(lists, num_shuffles)
    else:
        return grid_hyper_parameters_iterator(lists)


def save_json(obj, name):
    import json
    with open(name, "w") as f:
        json.dump(obj, f, skipkeys=True, indent=True)


def getter(analyzed_names, model_hyper_parameters, trial):
    analyzed_hyper_parameters = []
    for name in analyzed_names:
            analyzed_hyper_parameters.append(model_hyper_parameters[name])
    return tuple(analyzed_hyper_parameters + [trial])

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator