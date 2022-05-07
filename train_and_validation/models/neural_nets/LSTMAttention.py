import torch.nn as nn
import torch
from .berkley_transformer import AttentionLayer

class CustomLSTM2(nn.Module):
    def __init__(self, device, input_size, num_layers, hidden_size, dropout=0, bias=False):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bias=bias)
        torch.nn.init.zeros_(self.lstm.bias_hh_l0)
        torch.nn.init.zeros_(self.lstm.bias_ih_l0)
        self.lstm.weight_hh_l0.data.uniform_(-0.5,0.5)
        self.lstm.weight_ih_l0.data.uniform_(-0.5,0.5)
        # This hidden state and cell state are the last states that were forwarded,
        # this can be used in the validation in order to start from where you left.

    def forward(self, x_inp, segrold):
        # train_or_validation_or_test is a variable that controls between train, test or validation
        # Note that in the train mode the h0 and c0 are restarted to 0,
        # in the validation mode the h0 and c0 are the ones from last time.
        # x_inp should have dimensions (batch_size, seq_length, num_features)

        x_inp = x_inp.to(self.device).float()
        batch_size = x_inp.shape[0]

        new_out, segr = self.lstm(x_inp, segrold)

        final_out = new_out
        return final_out, segr
        # output should have dimensions: (batch_size, seq_length, hidden_size)

class CustomLSTMAttention(nn.Module):
    def __init__(self, device, input_size, num_layers, hidden_size, dropout=0, bias=False,
                    n_head=1, d_k=None, d_v=None):
        super().__init__()

        if d_k == None:
            d_k = hidden_size
        if d_v == None:
            d_v = hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = CustomLSTM2(device, input_size, num_layers, hidden_size, dropout, bias)
        self.attention = AttentionLayer(device, n_head, hidden_size, d_k, d_v, dropout)
        # This hidden state and cell state are the last states that were forwarded,
        # this can be used in the validation in order to start from where you left.

    def forward(self, x_inp):
        # x_inp should have dimensions (batch_size, seq_length, num_features)
        batch_size = x_inp.shape[0]
        segr = None
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        num_divisions = x_inp.shape[1]//500 + 1
        out_list = []
        for division in range(num_divisions):
            new_input = x_inp[:,division*500:(division+1)*500]
            if segr == None:
                segr = (h0, c0)
            final_out, segr = self.lstm(new_input, segr)
            out_list.append(self.attention(final_out))
        return torch.cat(out_list, dim=1)
        # output should have dimensions: (batch_size, seq_length, hidden_size)
