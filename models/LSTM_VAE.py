import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2
MAX_LENGTH = 10

class LSTM_VAE(nn.Module):
    def __init__(self, embedding_dim, rnn_hidden_size, rnn_num_layers):
        super(LSTM_VAE, self).__init__()
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                           batch_first=True)
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_dim = embedding_dim

    def forward(self, xo_embed, xo_len, h0=None, c0=None):
        packed_input = pack_padded_sequence(xo_embed, xo_len, batch_first=True, enforce_sorted=False)
        if h0 is None or c0 is None:
            h0, c0 = (torch.zeros(self.rnn_num_layers, len(xo_len), self.rnn_hidden_size).to(xo_embed.device),
                      torch.zeros(self.rnn_num_layers, len(xo_len), self.rnn_hidden_size).to(xo_embed.device))
        output, (hT, cT) = self.rnn(packed_input, (h0, c0))
        return output, (hT, cT)

    



