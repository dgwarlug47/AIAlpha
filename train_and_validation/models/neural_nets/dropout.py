import torch.nn as nn
import torch


class LockedDropout(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x, dropout=0.5):
        # x has dimensions (batch_size, seq_length, num_features)
        x = torch.transpose(x, 0, 1)
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.clone().detach() / (1 - dropout)
        mask = mask.expand_as(x)
        return torch.transpose(mask * x, 0, 1)


class DaviDropout(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, features, dropout):
        # features has dimensions (batch_size, seq_length, num_features)
        mask = torch.zeros(features.size()[2]).bernoulli_(1 - dropout).repeat(features.shape[0], features.shape[1], 1).to(self.device)
        return features * mask