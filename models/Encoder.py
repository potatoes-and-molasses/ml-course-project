import torch
import torch.nn as nn

from models.LSTM_VAE import LSTM_VAE


class Encoder(nn.Module):
    def __init__(self, embedding_dim, rnn_hidden_size, rnn_num_layers, z_dim):
        super(Encoder, self).__init__()
        self.ose = LSTM_VAE(embedding_dim, rnn_hidden_size, rnn_num_layers)
        self.pse = LSTM_VAE(embedding_dim, rnn_hidden_size, rnn_num_layers)
        self.lin = nn.Linear(rnn_hidden_size * rnn_num_layers, 2 * z_dim)
        self.z_dim = z_dim

    def forward(self, xo_embed, xp_embed, xo_len, xp_len):
        _, (hT, cT) = self.ose(xo_embed, xo_len)
        _, (_, cT) = self.pse(xp_embed, xp_len, h0=hT, c0=cT)
        cT = torch.cat([c for c in cT], 1)
        mulogvar = self.lin(cT)
        mu = mulogvar[:, :self.z_dim].contiguous()
        logvar = mulogvar[:, self.z_dim:].contiguous()
        return mu, logvar