
import torch.nn as nn
import torch


class CustomGRU(nn.Module):
    def __init__(self, device, input_size, num_layers, hidden_size, dropout=0, bias=False):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bias=bias)
        # This hidden state and cell state are the last states that were forwarded,
        # this can be used in the validation in order to start from where you left.
        self.last_h0 = None


    def forward(self, x_inp, train_or_validation_or_test):
        # train_or_validation_or_test is a variable that controls between train, test or validation
        # Note that in the train mode the h0 and c0 are restarted to 0,
        # in the validation mode the h0 and c0 are the ones from last time.
        # x_inp should have dimensions (batch_size, seq_length, num_features + 2)
        x_inp = x_inp.to(self.device).float()
        batch_size = x_inp.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        new_out, _ = self.gru(x_inp, h0)

        final_out = new_out
        return final_out
        # output should have dimensions: (batch_size, seq_length, hidden_size)
