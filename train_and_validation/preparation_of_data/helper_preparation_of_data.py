import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randrange


def loadTickers(fileName):
    # receives as input the address where the names of the tickers are stored.
    # outputs the list of the tickers names.
    lineList = list()
    with open(fileName) as f:
        for line in f:
            lineList.append(line)
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList


def discretization_of_real_returns(real_returns, d, lower_bound, upper_bound):
    # real_returns has dimensions (num_ data_points)
    # the real returns get converted to labels
    bins = np.linspace(lower_bound, upper_bound, d - 1)
    labels = np.digitize(real_returns, bins)
    return labels


def new_train_test_validation_data_creator(data, real_returns, data_preparation_hyper_parameters, info):
    # data should have dimensions (num_securities, num_data_points, num_features)
    # real_returns should have dimensions (num_securities, num_data_points)
    # formats the data according to the seq_len, and train_test_validation
    # returns train_data and test_data which should have dimensions (num_securities)
    assert(data.shape[1] == real_returns.shape[1]), "real returns and the data should have the same amount of data points"
    assert(data.shape[1] == len(info['all_days'])), "all days and the data should have the same amount of data points"
    test_begin = info["test_begin"]
    validation_begin = info["validation_begin"]

    try:
        index_test_begin = list(map(lambda i: i >= test_begin, info['all_days'])).index(True)
        index_validation_begin = list(map(lambda i: i >= validation_begin, info['all_days'])).index(True)
    except:
        raise "you selected days that were not available"

    train_data = data[:, :index_test_begin, :]
    train_returns = real_returns[:, :index_test_begin]
    info['train_days'] = info['all_days'][:index_test_begin]

    test_data = data[:, index_test_begin:index_validation_begin, :]
    test_returns = real_returns[:, index_test_begin:index_validation_begin]
    info['test_days'] = info['all_days'][index_test_begin:index_validation_begin]

    validation_data = data[:, index_validation_begin:, :]
    validation_returns = real_returns[:, index_validation_begin:]
    info['validation_days'] = info['all_days'][index_validation_begin:]

    return train_data, test_data, validation_data, train_returns, test_returns, validation_returns


def artificial_data_removal(data):
    # data has dimensions: num_securties, num_sequences, seq_length, num_features
    # the output has dimension (?,  seq_length, num_features) and it is cleaned (without artificial data).
    flattened_data = flatten_data(data)
    real_data = []
    for index in range(flattened_data.shape[0]):
        difference = (flattened_data[index, :, 1:] != (np.zeros([data.shape[2], data.shape[3] - 1]))).sum()
        if (difference != 0):
            real_data.append(flattened_data[index])
    real_data = np.array(real_data)
    flattened_data = real_data
    return flattened_data


def flatten_data(data):
    # flattens the data
    seq_len = data.shape[2]
    num_features = data.shape[3]
    return np.reshape(data, (-1, seq_len, num_features))


def store_and_show_true_distribution(data_preparation_hyper_parameters, data_clean, num_bins, title):
    # data has dimensions (num_sequences, seq_length, features)
    # num_bins is the number of bins the returns were quantized on.
    # this method simply plots and returns the distribution of the data provided..
    labels = data_clean[:, :, 0]
    total_num_of_data_points = labels.size
    true_distribution = []
    for i in range(num_bins):
        bin_frequency = len(labels[np.where(labels == i)])
        true_distribution.append(bin_frequency / total_num_of_data_points)
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_bins), true_distribution, color='indigo')
    ax.set_title(title)
    plt.show()
    # for index in range(len(true_distribution)):
    #    print('index', index, true_distribution[index])
    data_preparation_hyper_parameters[title] = np.array(true_distribution)


def save_mean_and_variance(flattened_data, info):
    # the dimensions of flattened_data should be  (num_tickers, num_days, num_of_features)
    # note that we will save the results in the info.
    # note that the first feature of the flattened_data is the actual label,
    # this will not be normalized. Therefore we do not save the labels average
    # nor its standard deviation.
    num_of_features = flattened_data.shape[2]
    thing = np.reshape(flattened_data, (-1, num_of_features))
    info['train_mean'] = torch.FloatTensor(np.mean(thing[:, 2:], axis=0))
    info['train_std'] = torch.FloatTensor(np.std(thing[:, 2:], axis=0))


class AssetsData():
    def __init__(self, data, seq_length, bootstrap_ratio, bootstrap_seed, device):
        # data (num_securities, num_of_days, num_of_features + 2)
        self.data = torch.tensor(data).to(device)
        self.seq_length = seq_length
        self.bootstrap_ratio = bootstrap_ratio
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap = self.bootstrap_ratio != 0
        self.num_sequences = self.data.shape[1] - self.seq_length - 1
        if self.bootstrap:
            self.probability_distribution_index = []
            if self.bootstrap_seed is None:
                while len(self.probability_distribution_index) < self.bootstrap_ratio*self.num_sequences - 1:
                    self.probability_distribution_index.append(randrange(self.num_sequences))
            else:
                random.seed(bootstrap_seed)
                bins = np.linspace(0, 1, self.num_sequences)
                while len(self.probability_distribution_index) < self.bootstrap_ratio*self.num_sequences - 1:
                    key = random.random()
                    index = np.digitize([key], bins)[0]
                    self.probability_distribution_index.append(index)
    def getitem(self):
        if not (self.bootstrap):
            index = randrange(self.num_sequences)
        else:
            index = self.probability_distribution_index[randrange(len(self.probability_distribution_index))]
        return self.data[:, index:index+self.seq_length]

def remove_trash(my_array):
    my_array[my_array == -np.inf] = 0
    my_array[my_array == np.inf] = 0
    my_array[my_array != my_array] = 0
    return my_array