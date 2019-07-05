import torch
import torch.nn as nn

from models.LSTM_VAE import LSTM_VAE

class Decoder(nn.Module):
    def __init__(self, embedding_dim, rnn_hidden_size, rnn_num_layers, dictionary_size):
        super(Decoder, self).__init__()
        self.ose = LSTM_VAE(embedding_dim, rnn_hidden_size, rnn_num_layers)
        self.pse = LSTM_VAE(rnn_hidden_size + embedding_dim, rnn_hidden_size, rnn_num_layers)
        self.linear = nn.Linear(rnn_hidden_size, dictionary_size)

    def forward(self, xo_embed, z, xo_len, z_len):
        
        _, (hT, cT) = self.ose(xo_embed, xo_len)
        out = hT[-1]

        final = []
        for i in range(z_len.max().item()):
            real_inp = torch.cat((z, out), 1).unsqueeze(1).contiguous() #concat Z and the output of the previous step

            output, (hT, cT) = self.pse(real_inp, torch.tensor([1] * len(z)), h0=hT, c0=cT) #performing one step in the LSTM
            out = hT[-1]

            final.append(out)

        final = torch.stack(final, 1) #tensor of all the outputs of all the cells of the LSTM
        logits = self.linear (final) #the output vector on which the softmax+cross-entropy is enacted
        return logits