import time
import torch
import torch.nn as nn
import numpy as np
import random
from .helper_validation import save_performance_metrics
from captum.attr import IntegratedGradients
from captum.attr import FeatureAblation


def rnn_execution(model, data, next_day_real_returns):
    # gets the positions respective of the data
    # model_hyper_parameters was already explained in the docs folder.
    # next_day_real_returns has dimensions (num_of_securities, num_days)
    time6 = time.process_time()
    with torch.no_grad():
        model.mode_toggle('validation')
        pos = (model.get_position_from_input(data).cpu().detach().numpy())
        model.leverage_toggle(False)
        unnormalized_pos = (model.get_position_from_input(data)).cpu().detach().numpy()
        model.leverage_toggle(True)
        pnl = next_day_real_returns * pos

    time7 = time.process_time()
    print(time7 - time6, 'this was the total time of the rnn execution')
    return np.array(next_day_real_returns), np.array(pos), np.array(pnl), np.array(unnormalized_pos)

"""
def feature_importance(model, data, minutes):
    if minutes == 0:
        return 0, 0
    with torch.no_grad():
        previous = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        model.mode_toggle('validation')
        model.leverage_toggle(False)
        ig = IntegratedGradients(model)
        features_importances = 0
        rolled_seq_length = 252
        time8 = time.process_time()
        target = rolled_seq_length-1
        time9 = time8
        counter = 0
        while (time9 - time8) < minutes * 60:
            counter = counter + 1
            day_index = random.randrange(data.shape[1] - (rolled_seq_length + 1))
            new_input = data[:, day_index:(day_index + rolled_seq_length)]
            attributions = ig.attribute(new_input, target=target)
            features_importances = features_importances + (abs(attributions)).sum(axis=0).sum(axis=0)
            zeros = torch.zeros(new_input.shape)
            out_zeros = model(zeros)
            out_x = model(new_input)
            diff = out_x - out_zeros
            ras = diff[:,target]
            sas = attributions.sum(axis=-1).sum(axis=-1)
            time9 = time.process_time()
        torch.backends.cudnn.enabled = previous
    average_time = (time9 - time8)/counter
    features_importances = features_importances[2:].cpu().detach().numpy()/counter
    print("average time of feature importance", average_time)
    print("number of targets", counter)
    baseline_position = out_zeros[0,-1]
    model.leverage_toggle(True)
    return features_importances, baseline_position
"""

class WrapperForAttribution(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = model.device
        self.model = model

    def forward(self, x_inp):
        # x_inp should have dimension
        # (n_step, num_tickers, seq_length, num_features)
        x_inp = x_inp.to(self.device).float()
        n_steps = x_inp.shape[0]
        new_out_list = []
        for n_step in range(n_steps):
            new_out = self.model(x_inp[n_step])
            new_out_list.append(new_out.unsqueeze(dim=0))
        all_out = torch.cat(new_out_list)
        return all_out


def feature_importance2(model, data, info):
    # data has dimensions (num_tickers, all_days, num_features)
    minutes = info['feature_importance_minutes']
    if minutes == 0:
        return 0, 0

    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    model.leverage_toggle(False)
    model.mode_toggle('validation')
    wrapper = WrapperForAttribution(model)
    method_type = info['feature_importance_method']
    if method_type == 'IntegratedGradients':
        ig = IntegratedGradients(wrapper)
    elif method_type == 'FeatureAblation':
        ig = FeatureAblation(wrapper)
    else:
        raise "this method for feature importance is not available yet"
    with torch.no_grad():
        features_importances = 0
        lags_importances = 0
        tickers_importances = 0
        rolled_seq_length = 252
        num_tickers = data.shape[0]
        time8 = time.process_time()
        time9 = time8
        counter = 0
        while (time9 - time8) < minutes * 60:
            counter = counter + 1
            now_ticker = random.randrange(num_tickers)
            target = (now_ticker, rolled_seq_length - 1)
            day_index = random.randrange(data.shape[1] - (rolled_seq_length + 1))
            new_input = data[:, day_index:(day_index + rolled_seq_length)]
            attributions = (ig.attribute(  (new_input).unsqueeze(0), baselines=model.get_baselines(rolled_seq_length), target=target)).squeeze(0)
            # attributions has dimensions (num_tickers, seq_length, num_features)
            features_importances = features_importances + (abs(attributions)).sum(axis=0).sum(axis=0)
            lags_importances = lags_importances + (abs(attributions)).sum(axis=0).sum(axis=1)
            tickers_importances = tickers_importances + (abs(attributions)).sum(axis=1).sum(axis=1)
            zeros = torch.zeros(new_input.shape).to(model.device)
            out_zeros = wrapper(zeros.unsqueeze(0))
            out_x = wrapper(new_input.unsqueeze(0))
            diff = out_x - out_zeros
            ras = diff[:, target[0], target[1]]
            sas = attributions.sum(axis=-1).sum(axis=-1)
            time9 = time.process_time()
    torch.backends.cudnn.enabled = previous
    average_time = (time9 - time8)/counter
    features_importances = features_importances[2:].cpu().detach().numpy()/counter
    lags_importances = lags_importances.cpu().detach().numpy()/counter
    tickers_importances = tickers_importances.cpu().detach().numpy()/counter
    print("average time of attributions compuatation", average_time)
    print("number of targets", counter)
    baseline_position = out_zeros[0, 0, -1].item()
    importances_dict = {}
    importances_dict['baseline_position'] =  baseline_position
    importances_dict['number of targets'] = counter
    importances_dict['average time of attributions computation'] = average_time
    model.leverage_toggle(True)
    return [features_importances, lags_importances, tickers_importances], importances_dict


def new_rnn_train_test_validation_execution_collect_performance_metrics(model, data_sets, model_hyper_parameters, info):
    # validates the model in both the validation_data_sets, collects cumulative data
    # data_preparation_hyper_parameters was already explained in the docs folder.
    # model_hyper_parameters was already explained in the docs folder.
    # real_returns has dimensions (validation_num_of_securities, num_days)
    all_data = torch.cat((data_sets[0][0], data_sets[1][0], data_sets[2][0]), axis=1)
    all_next_day_real_returns = np.concatenate((data_sets[0][1], data_sets[1][1], data_sets[2][1]), axis=1)

    _, all_pos, all_pnl, all_unnormalized_pos = rnn_execution(model, all_data, all_next_day_real_returns)

    num_train_days = data_sets[0][0].shape[1]
    num_test_days = data_sets[1][0].shape[1]

    pos_list = list_creator(all_pos, num_train_days, num_test_days)
    pnl_list = list_creator(all_pnl, num_train_days, num_test_days)
    unnormalized_pos_list = list_creator(all_unnormalized_pos, num_train_days, num_test_days)

    importances, importances_dict = (feature_importance2(model, data_sets[2][0], info))

    return save_performance_metrics([pos_list, pnl_list, unnormalized_pos_list, importances, importances_dict], info)


def list_creator(x, num_train_days, num_test_days):
    x_list = []

    x_list.append(x[:, :num_train_days])
    x_list.append(x[:, num_train_days:(num_train_days + num_test_days)])
    x_list.append(x[:, (num_train_days + num_test_days):])

    return x_list
