# docs complete.
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .neural_nets.LSTM import CustomLSTM
from .neural_nets.GRU import CustomGRU
from .neural_nets.TCN import CustomTCN
from .neural_nets.Linear import CustomLinear
from .neural_nets.berkley_transformer import Transformer
from .neural_nets.LSTMAttention import CustomLSTMAttention
from .neural_nets.dropout import DaviDropout, LockedDropout


class Net(nn.Module):
    def __init__(self, model_hyper_parameters, info):
        super().__init__()
        """ configuration of the input """
        self.cross_sectional = model_hyper_parameters['cross_sectional']
        self.num_tickers = info['num_tickers']

        if self.cross_sectional:
            self.input_channels = info['num_features'] * self.num_tickers
        else:
            self.input_channels = info['num_features']

        self.device = info['device']
        """ configuration of the neural net """
        self.bias = model_hyper_parameters['bias']
        self.hidden_size = model_hyper_parameters['hidden_size']
        self.n_head = model_hyper_parameters['n_head']
        self.d_k = model_hyper_parameters['d_k']
        self.d_v = model_hyper_parameters['d_v']
        self.num_layers = model_hyper_parameters['num_layers']
        self.type_net = model_hyper_parameters['type_net']

        """ configuration of dropout """
        self.dropout_value = model_hyper_parameters['dropout']
        self.type_dropout = model_hyper_parameters['type_dropout']
        if self.type_dropout == "Davi":
            self.Dropout = DaviDropout(self.device)
        elif self.type_dropout == "Locked":
            self.Dropout = LockedDropout(self.device)
        else:
            raise "this option of Dropout is not available yet"
        self.dropout_on = True

        """ Selecting the type of neural network. """
        if self.type_net == 'LSTM':
            self.base_net = CustomLSTM(self.device, input_size=self.input_channels, num_layers=self.num_layers, hidden_size=self.hidden_size, bias=self.bias)
            self.net_output_channels = self.hidden_size
        elif self.type_net == 'Linear':
            self.base_net = CustomLinear(self.device, input_channels=self.input_channels, d=self.hidden_size, bias=self.bias)
            self.net_output_channels = self.hidden_size
        elif self.type_net == 'GRU':
            self.base_net = CustomGRU(self.device, input_size=self.input_channels, num_layers=self.num_layers, hidden_size=self.hidden_size, bias=self.bias)
            self.net_output_channels = self.hidden_size
        elif self.type_net == 'TCN':
            self.base_net = CustomTCN(self.device, self.input_channels, [self.hidden_size for _ in range(self.num_layers)], dropout=self.dropout_value)
            self.net_output_channels = self.hidden_size
        elif self.type_net == 'LSTMAttention':
            self.base_net = CustomLSTMAttention(self.device, self.input_channels, self.num_layers, self.hidden_size, self.dropout_value, self.bias,
                    self.n_head, self.d_k, self.d_v)
            self.net_output_channels = self.hidden_size
        else:
            raise "this option of neural net is not provided yet"

        """ Selecting the linear layer """
        if not (self.cross_sectional):
            self.linear_mapping_for_dimensionality_adjustment = torch.nn.Linear(self.net_output_channels, 1, bias=self.bias)
        else:
            self.linear_mapping_for_dimensionality_adjustment = torch.nn.Linear(self.net_output_channels, self.num_tickers, bias=self.bias)

        """ input_normalization part """
        self.input_normalization = model_hyper_parameters['input_normalization']
        if self.input_normalization:
            self.normalization_scalar = torch.nn.Parameter(torch.ones([1]))
            self.normalization_bias = torch.nn.Parameter(torch.zeros([1]))
        self.train_features_mean = info['train_mean'].to(self.device)
        self.train_features_std = info['train_std'].to(self.device)

    def forward(self, features):
        # features has dimensions (num_tickers, seq_length, num_features)
        """ Taking care of the tuned input normalization """
        if self.input_normalization:
            features = (features - self.train_features_mean) / self.train_features_std
            features = features * self.normalization_scalar + self.normalization_bias
        """ input dropout layer """
        if self.dropout_on:
            features = self.Dropout(features, dropout=self.dropout_value)

        """ Forward propagate neural network """
        if not(self.cross_sectional):
            new_input = features
        else:
            new_input = features.transpose(0, 1).reshape(features.shape[1], features.shape[0] * features.shape[2]).unsqueeze(0)

        output = self.base_net(new_input)
        # if cross sectional output should have dimensions (1, seq_length, hidden_size)
        # if not cross sectinal output should have dimensions (num_tickers, seq_length, hidden_size, num_tickers)

        """ output dropout layer """
        if self.dropout_on:
            output = self.Dropout(output, dropout=self.dropout_value)

        final_output = self.linear_mapping_for_dimensionality_adjustment(output)

        if self.cross_sectional:
            # in this case final_output should have dimensions: (1, seq_length, num_tickers)
            final_output = final_output.transpose(0, 2).squeeze(2)
        else:
            final_output = final_output.squeeze(2)
        return final_output


class Deep_Model(nn.Module):
    def __init__(self, model_hyper_parameters, info):
        super().__init__()
        """ configuration of the input """
        self.device = info['device']
        self.num_tickers = info['num_tickers']
        self.batch_size = model_hyper_parameters['batch_size']

        self.loss_function_from_returns = model_hyper_parameters['loss_function_from_returns']
        self.shrinkage = model_hyper_parameters['shrinkage']

        """ last layer """
        self.last_layer_funcs = []
        if model_hyper_parameters['last_layer_type'] == 'Tanh':
            self.last_layer_funcs.append(nn.Tanh())
        elif model_hyper_parameters['last_layer_type'] == 'Softmax':
            self.last_layer_funcs.append(nn.Softmax(dim=0))
        elif model_hyper_parameters['last_layer_type'] == 'Softsign':
            self.last_layer_funcs.append(nn.Softsign())
        elif model_hyper_parameters['last_layer_type'] == 'Sigmoid':
            self.last_layer_funcs.append(nn.Sigmoid())
        elif model_hyper_parameters['last_layer_type'] == 'None':
            def id(x):
                return x
            self.last_layer_funcs.append(id)
        else:
            raise "this last layer is not provided yet"
        
        """ max allocation """
        self.max_allocation = info['max_allocation']
        if self.max_allocation == 'None':
            print('no max allocation')
        else:
            def place_holder77(x):
                return clip_allocation(x, self.max_allocation)
            self.last_layer_funcs.append(place_holder77)
        def total_last_layer(x):
            for last_layer_func in self.last_layer_funcs:
                x = last_layer_func(x)
            return x
        self.last_layer = total_last_layer

        """ transaction costs"""
        self.train_transaction_cost_factor = model_hyper_parameters['train_transaction_cost_factor']
        self.test_transaction_cost = torch.from_numpy(info['test_transaction_cost']/10000).to(self.device)
        self.train_transaction_cost = self.train_transaction_cost_factor * self.test_transaction_cost
        self.transaction_cost = self.train_transaction_cost

        """ benchmark """
        self.benchmark = info['long_only_benchmark']

        """ leverage """
        self.leverage = True

        """ net """
        self.net = Net(model_hyper_parameters, info)

        """ using_swa """
        self.using_swa = model_hyper_parameters['using_swa']

        """ swa net """
        swa_decay = model_hyper_parameters['swa_decay']
        if swa_decay < 1e-3:
            self.swa_net = torch.optim.swa_utils.AveragedModel(self.net)
        else:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - swa_decay) * averaged_model_parameter + swa_decay * model_parameter
            self.swa_net = torch.optim.swa_utils.AveragedModel(self.net, avg_fn=ema_avg)
        self.on_swa = False

    def get_position_from_input(self, x):
        # x has dimensions (batch_size, seq_len, n_features + 2)
        # x[;, ;, 0] is the labels and/or returns
        # x[:, :, 1] is the leverage
        # x[:, :, 2:] is the features.
        # the output is a float the actual loss.
        # train_or_validation is a string which is either: 'train' or 'validation' or 'test'
        # features has dimensions: (bath_size, seq_len, num_features)
        # returns the neural_net_output of the features.
        x = x.to(self.device)
        features = x[:, :, 2:]
        features = features.clone()
        if self.on_swa:
            final_output = self.swa_net(features)
        else:
            final_output = self.net(features)

        # in both cases, the cross sectional and the univariate, the final_output should have dimensions
        # (num_tickers, seq_length)
        if not self.benchmark:
            position = self.last_layer(final_output)
        else:
            position = torch.ones(x.shape[:-1]).to(self.device)

        if self.leverage:
            leverage_factor = x[:, :, 1]
            return position * leverage_factor
        else:
            return position

    def get_returns_from_input(self, x, epoch=0):
        # x has dimensions (batch_size, seq_len, n_features + 2)
        # x[;, ;, 0] is the labels or returns
        # x[;, ;, 1] is the leverage
        # x[:, :, 2:] is the features.
        # the output is a float the actual loss.
        returns = x[:, :, 0]
        position = self.get_position_from_input(x)
        shifted_position = position[:,:-1]
        shifted_position = torch.cat((torch.zeros(position.shape[0], 1).to(self.device), shifted_position), dim=1).to(self.device)
        portifolio_returns = ((position*returns - (self.transaction_cost.unsqueeze(1))*(torch.abs(position-shifted_position))).transpose(0,1)).sum(-1)
        return portifolio_returns

    def get_loss_from_returns(self, returns):
        # returns should have dimensions (num_days)
        """ choosing appropriate loss function """
        if self.loss_function_from_returns == 'sharpe':
            pure_loss = - 15.87*((returns.mean())/(returns.std()))
        elif self.loss_function_from_returns == 'fake_sharpe':
            pure_loss = - ((returns.mean())/ ((returns.std())**2))
        elif self.loss_function_from_returns == 'kelly':
            pure_loss = - kelly(returns)
        elif self.loss_function_from_returns == 'kelly2':
            pure_loss = - kelly2(returns)
        elif self.loss_function_from_returns == 'average':
            pure_loss = - returns.mean()
        elif self.loss_function_from_returns == 'sortino':
            pure_loss = - 15.87*returns.mean()/((returns[returns < 0]).std())
        else:
            raise "this loss function is not provided"
        reg = self.shrinkage
        L1 = torch.nn.SmoothL1Loss(beta=1e-5)

        flattened_parameters = parameter_flattening(self)
        if self.device == torch.device('cuda'):
            zeros = torch.zeros(list(flattened_parameters.shape), requires_grad=True).cuda()
        else:
            zeros = torch.zeros(list(flattened_parameters.shape), requires_grad=True)
        check = pure_loss + reg*L1(flattened_parameters, zeros)
        if check.item() != check.item():
            print('We found NaN')
            raise "We have found NaN"
        return check

    def forward(self, x):
        return self.get_position_from_input(x)

    def mode_toggle(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.net.dropout_on = True
            self.transaction_cost = self.train_transaction_cost
            self.on_swa = False
            self.train()
        elif self.mode == "test":
            self.net.dropout_on = False
            self.transaction_cost = self.test_transaction_cost
            if self.using_swa:
                self.on_swa = True
                self.swa_net.module.dropout_on = False
            else:
                self.on_swa = False
            self.eval()
        elif self.mode == "validation":
            self.net.dropout_on = False
            self.transaction_cost = 0
            if self.using_swa:
                self.on_swa = True
                self.swa_net.module.dropout_on = False
            else:
                self.on_swa = False
            self.eval()
        else:
            raise "this mode is not currently available"

    def leverage_toggle(self, bool_value):
        self.leverage = bool_value

    def update_swa(self):
        self.swa_net.update_parameters(self.net)

    def get_baselines(self, rolled_seq_length):
        if self.net.input_normalization:
            if self.on_swa:
                the_net = self.swa_net
            else:
                the_net = self.net
            life = (the_net.train_features_mean - (the_net.normalization_bias*the_net.train_features_std)/the_net.normalization_scalar)
            life = torch.cat((torch.ones([2]).to(self.device), life))
            return life.unsqueeze(0).unsqueeze(0).repeat(self.num_tickers, rolled_seq_length, 1).unsqueeze(0)
        else:
            return None


def clip_allocation(pos, max_allocation):
    # pos has dimensions (num_tickers, seq_length)
    num_tickers = pos.shape[0]
    clip_pos = torch.clamp(pos, max=max_allocation)
    diff = pos - clip_pos
    diff = diff.sum(axis=0)
    diff = diff.unsqueeze(0).repeat(num_tickers,1)/num_tickers
    final_pos = clip_pos + diff
    return final_pos


def kelly(PNL):
     s = torch.prod(1.0+torch.clamp(PNL, min=-0.9))
     return torch.log(s)


def kelly2(PNL):
     s = torch.prod(1.0+torch.clamp(5*PNL, min=-0.9))
     return torch.log(s)


def parameter_flattening(model):
    new_params = torch.zeros([0]).to(model.device)
    for _, param in model.named_parameters():
        if param.requires_grad:
            new_params = torch.cat((new_params, param.data.clone().reshape(-1)))
    return new_params