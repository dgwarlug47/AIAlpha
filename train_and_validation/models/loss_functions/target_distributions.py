# doc complete.
import numpy as np
import torch


def target_distributions_generator(d, distribution_name, distribution_parameter):
    # Disclaimer: in case you set the distribution_name as 'One_Hot', this method
    # will not return an artificial distribtuion, it will actually return the observed distributiion
    # which is ( 0, 0, 0, ......, 1, ..... 0)

    # d is the number of bins.
    # distribution_name is the name of the probability distribution you want to use
    # the output of this method is of dimensions: (d, d),
    # that is the distribution created for each possible label.
    label = np.arange(d)
    target_distribution_from_all_labels = []
    for index in range(label.shape[0]):
        if distribution_name == 'One_Hot':
            target_distribution_from_one_label = np.zeros(d)
            target_distribution_from_one_label[label[index].item()] = 1.
        else:
            if distribution_name == 'Normal':
                from scipy.stats import norm
                distribution = norm(loc=label[index].item(), scale=distribution_parameter)
            if distribution_name == 'Laplace':
                from scipy.stats import laplace
                distribution = laplace(loc=label[index].item(), scale=distribution_parameter)
            if distribution_name == 'Cauchy':
                from scipy.stats import cauchy
                distribution = cauchy(loc=label[index].item(), scale=distribution_parameter)
            if distribution_name == 'Logistic':
                from scipy.stats import logistic
                distribution = logistic(loc=label[index].item(), scale=distribution_parameter)
            if distribution_name == 'Student':
                from scipy.stats import t
                distribution = t(loc=label[index], df=distribution_parameter)
            target_distribution_from_one_label = np.array([])
            for i in range(d):
                target_distribution_from_one_label = np.append(target_distribution_from_one_label, (distribution.cdf(i + 0.5) - distribution.cdf(i - 0.5)))
        target_distribution_from_all_labels.append(target_distribution_from_one_label / sum(target_distribution_from_one_label))
    return np.array(target_distribution_from_all_labels)


def target_distributions_getter(target_distributions, device, labels):
    # labels should have dimensions (num_targets)
    # target_distributions has dimensions (d, d) the distribution for each label.
    # returns the actual target_distributions of each for each label specified in labels,
    # the dimension of the output is (num_targets, labels)
    return torch.FloatTensor(np.take(target_distributions, labels.cpu().numpy(), axis=0)).to(device)
