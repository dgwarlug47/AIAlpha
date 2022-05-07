# ################    Self-Attention    #####################
# from https://github.com/jadore801120/attention-is-all-you-need-pytorch/

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=784):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) * 0.1

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, device, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.device = device
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x has dimensions (batch_size, seq_len, d_in)

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, device, temperature, attn_dropout=0.0):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # I am not entirely sure but I think that the dimensions of
        # q, k, v are (batch_size, n_heads, seq_length, d_)
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3).contiguous())
        # attn has dimensions (batch_size, n_heads, seq_length, seq_length)

        if mask is not None:
            attn = attn.masked_fill((mask == 0), -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        # output has dimesions (batch_size, n_heads, seq_length, d_v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, device, n_head, d_model, d_k, d_v, dropout=0.1):
        # d_something in this context stands for the dimension
        # of somehitng.
        # k stands for keys, q query and v values.
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.device = device

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(self.device, temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # q, k, v all have dimension (batch_size, seq_length, d_model)
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # sz_b stands for size batch (I think).
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # For head axis broadcasting.
            mask = mask.unsqueeze(0).unsqueeze(0)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads
        # together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q


class AttentionLayer(nn.Module):
    def __init__(self, device, n_head, d_model, d_k, d_v, dropout=0.):
        super(AttentionLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(device, n_head, d_model, d_k, d_v, dropout=dropout)
        self.device = device

    def forward(self, dec_input):
        # dec_input should have dimensions (batch_size, seq_length, num_features)
        # for reasons I don't know why in the next line of code,
        # dec_input is all three query, values and keys. (Weird)

        """ register_buffer creates a torch variable that will not """
        """ be tracked by the optimizer I think. """
        seq_length = dec_input.shape[1]
        mask = torch.zeros(seq_length, seq_length).to(self.device)
        for i in range(seq_length):
            mask[i, :(i + 1)] = 1
        dec_output = self.slf_attn(dec_input, dec_input, dec_input, mask=mask)
        return dec_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, device, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(device, n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(device, d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, mask=None):
        # for reasons I don't know why in the next line of code,
        # dec_input is all three query, values and keys. (Weird)
        dec_output = self.slf_attn(dec_input, dec_input, dec_input, mask=mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class Transformer(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, device, d, seq_length, input_size, n_layers, d_model, n_heads, d_inner, d_k=16, d_v=64, dropout=0, mode='none'):
        super().__init__()
        self.seq_length = seq_length
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.input_size = input_size
        self.mode = mode
        self.device = device
        self.d = d
        self.fc_final = nn.Linear(self.d_model, self.d, bias=False)

        if mode == 'pos_encoding':
            self.pos_enc = PositionalEncoding(1, n_position=784)
        self.fc_in = nn.Linear(self.input_size, d_model)
        # array of  decoders, where each decoder is a "layer".
        self.layer_stack = nn.ModuleList([
            DecoderLayer(device=self.device, d_model=self.d_model, d_inner=self.d_inner,
                         n_head=self.n_heads, d_k=self.d_k, d_v=self.d_v, dropout=self.dropout)
            for _ in range(self.n_layers)])

        # register_buffer creates a torch variable that will not
        # be tracked by the optimizer I think.
        self.register_buffer('mask', torch.zeros(self.seq_length, self.seq_length))
        self.mask = self.mask
        # self.mask get intialized as an upper triangular matrix.
        # with only ones in the below part.
        for i in range(self.seq_length):
            self.mask[i, :(i + 1)] = 1

    def forward(self, x, train_or_validation):
        # if mode = none, x has dimensions (batch_size, seq_length, input_size + 1)
        x = x.to(self.device)
        x = x[:, :, 1:]
        x = x.float()
        assert(self.seq_length == x.shape[1]), ("x has seq_length " + str(x.shape[1]) + 'should have seq_length ' + str(self.seq_length))

        batch_size = x.shape[0]
        if self.mode == 'pixel_location':
            # x = append_location(x, self.device)
            x = x.permute(0, 2, 3, 1).view(batch_size, 784, self.input_size)
        elif self.mode == 'pos_encoding':
            x = x.view(batch_size, 784, self.input_size)
            x = self.pos_enc(x)
        if train_or_validation == 'train' or train_or_validation == 'validation':
            x = torch.cat((torch.zeros(batch_size, 1, self.input_size).to(self.device), x[:, :-1]), dim=1)
        # -- Forward
        x = F.relu(self.fc_in(x))
        for i, dec_layer in enumerate(self.layer_stack):
            x = dec_layer(x, mask=self.mask)
        x = x.view(batch_size, self.seq_length, -1)
        return self.fc_final(x)

    def nll(self, x):
        logits = self(x)
        return F.binary_cross_entropy_with_logits(logits, x)

    def sample(self, n):
        samples = torch.zeros(n, 1, 28, 28).to(self.device)
        with torch.no_grad():
            for r in range(28):
                for c in range(28):
                    logits = self(samples)[:, :, r, c]
                    probs = torch.sigmoid(logits)
                    samples[:, :, r, c] = torch.bernoulli(probs)
        return samples.cpu()