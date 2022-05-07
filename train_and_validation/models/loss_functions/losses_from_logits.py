# doc complete.
import torch
from .wasserstein import SinkhornDistance


def entropy(probs):
    # probs has dimensions: [num_distributions, d]
    # returns the entropy of each distribution.
    from torch.distributions import Categorical
    return Categorical(probs=probs).entropy()


def real_cross_entropy(logits_probs, target_distributions, eps):
    # logits_probs have dimension [batch, d]
    # target_distributions has dimension [batch, d] as well.
    # this is exaclty what the cross entropy does
    # returns a float which is the loss.
    log_probs = torch.nn.functional.log_softmax(logits_probs, dim=1)
    cross_entropies = - (log_probs * target_distributions).sum(-1)
    regularization = eps * entropy(torch.softmax(logits_probs, dim=1)).mean()
    return cross_entropies.mean() - regularization


def one_hot_wasserstein(device, probs, label, eps):
    # useless, did not work very well.
    label = label.float()
    batch_size = probs.shape[0]
    d = probs.shape[1]
    z = abs(torch.arange(d).unsqueeze(0).repeat(
        repeats=[batch_size, 1]).to(device) - label.unsqueeze(1))
    regularization = eps * entropy(probs).mean()
    return (probs * z).sum(axis=1)[0] - eps * entropy(probs) - regularization


def jensen_shannon_entropy(device, logits_probs, target_distributions, eps):
    # logits_probs has dimensions: [num_distributions, d]
    # target_distributions has dimensions: [num_distributions, d]
    # returns a float with the actual loss.
    new_logits_probs = torch.nn.functional.log_softmax(logits_probs, dim=1)
    probs = torch.softmax(new_logits_probs, dim=1)
    from torch.nn import KLDivLoss
    KLD = KLDivLoss(reduce=False)
    M = (1 / 2) * (target_distributions + probs)
    KL1 = KLD(new_logits_probs, M).sum(axis=1)
    KL2 = KLD(torch.log_softmax(target_distributions.float(),
                                dim=1), M).sum(axis=1)
    regularization = eps * entropy(torch.softmax(logits_probs, dim=1)).mean()
    return torch.sqrt((1 / 2) * (KL1 + KL2)) - regularization


def wasserstein(device, prob, target_distributions, d, eps):
    # prob has dimensions: [num_distributions, d]
    # target_distributions has dimensions: [num_distributions, d]
    # returns a float with the actual loss.
    def fitter(data):
        fit_data = data.unsqueeze(2).float()
        indexes = torch.arange(d).unsqueeze(0).unsqueeze(
            2).repeat([fit_data.shape[0], 1, 1]).to(device).float()
        return torch.cat((indexes, fit_data), dim=2)

    sinkhorn = SinkhornDistance(eps=eps, max_iter=50, reduction=None)
    dist, P, C = sinkhorn(fitter(prob).to(device),
                          fitter(target_distributions).to(device))
    return dist
