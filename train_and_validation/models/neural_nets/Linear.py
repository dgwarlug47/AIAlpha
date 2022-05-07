
import torch.nn as nn
import torch


class CustomLinear(nn.Module):
    def __init__(self, device, input_channels, d, bias):
        super().__init__()
        self.device = device
        self.input_channels = input_channels
        self.fc_out = nn.Linear(self.input_channels, d, bias=bias)

    def forward(self, x_inp, train_or_validation_or_test):
        x_inp = x_inp.to(self.device).float()
        return self.fc_out(x_inp)
        # output should have dimensions: (batch_size, seq_length, d)