import torch.nn as nn
import torch


class BetterLSTM(nn.Module):
    def __init__(self, device, input_size, num_layers, hidden_size, dropout):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        input_states_size = [input_size] + [hidden_size] * (num_layers - 1)
        self.norm = False
        self.lstm_layers = [nn.LSTM(input_size=input_states_size[index],
                                    hidden_size=hidden_size, dropout=dropout,
                                    batch_first=True,
                                    return_state_sequence=True).to(self.device) for index in range(num_layers)]
        for layer in self.lstm_layers:
            layer.requires_grad_(True)

        # This hidden state and cell state are the last states that were forwarded,
        # this can be used in the validation in order to start from where you left.
        self.last_h0 = None
        self.last_c0 = None

    def forward(self, x_inp, train_or_validation):
        x_inp = x_inp.to(self.device)
        # train_or_validation is a variable that controls between train or validation
        # in case it is train, the input gets some zeros concatenated in the begining,
        # and after that it is forwarded to the neural net.
        # in the validation mode, the input is forwarded to the neural net without changing anything
        # Note as well that in the train mode the h0 and c0 are restarted,
        # in the validation mode the h0 and c0 are the ones use the last time.
        # x_inp should have dimensions (batch_size, seq_length, num_features)

        batch_size = x_inp.shape[0]
        seq_length = x_inp.shape[1]
        num_of_features = x_inp.shape[2]
        if train_or_validation == 'train':
            self.last_h0 = None
            self.last_c0 = None
            h0 = torch.zeros(len(self.lstm_layers), batch_size, self.hidden_size).to(self.device)
            c0 = torch.zeros(len(self.lstm_layers), batch_size, self.hidden_size).to(self.device)
            x_inp = torch.cat((torch.zeros(batch_size, 1, num_of_features).to(self.device), x_inp[:, : - 1, :]), dim=1).to(self.device)
        else:
            if self.last_h0 is None:
                h0 = torch.zeros(len(self.lstm_layers), batch_size, self.hidden_size).to(self.device)
                c0 = torch.zeros(len(self.lstm_layers), batch_size, self.hidden_size).to(self.device)
            else:
                h0 = self.last_h0.clone()
                c0 = self.last_c0.clone()

        if self.norm:
            h0 = h0.half()
            c0 = c0.half()
            x_inp = x_inp.half()
            self.haste_half()

        final_h = torch.zeros(len(self.lstm_layers), batch_size, self.hidden_size).to(self.device)
        final_c = torch.zeros(len(self.lstm_layers), batch_size, self.hidden_size).to(self.device)
        final_out = torch.zeros(batch_size, seq_length, self.hidden_size).to(self.device)

        new_out, (new_input_h, new_input_c) = self.lstm_layers[0](x_inp, state=(h0[[0]], c0[[0]]))
        # since the lstm_layers have return_sequence_state = True, the dimensions of new_input_h and new_input_c = (1, batch_size, seq_length, hidden_size)
        final_h[0, :, :] = new_input_h[0, :, -1, :]
        final_c[0, :, :] = new_input_c[0, :, -1, :]
        for i in range(self.num_layers - 1):
            new_out, (new_input_h, new_input_c) = self.lstm_layers[i + 1](new_input_h[0, :, :, :], state=(h0[[i + 1]], c0[[i + 1]]))
            final_h[i + 1, :, :] = new_input_h[0, :, -1, :]
            final_c[i + 1, :, :] = new_input_c[0, :, -1, :]

        final_out = new_out
        if train_or_validation == 'validation':
            self.last_h0 = final_h
            self.last_c0 = final_c
        return final_out
        # output should have dimensions: (batch_size, seq_length, hidden_size)
