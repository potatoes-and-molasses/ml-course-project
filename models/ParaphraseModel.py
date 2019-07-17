import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.normal import Normal

from models.Encoder import Encoder
from models.Decoder import Decoder

SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2
MAX_PARA_LENGTH = 15


class ParaphraseModel(nn.Module):
    
    ''' 
        dicstionary_size = the size of the dictionary in the dataset
        embedding_dim = each word in the dictionary is embedded in a vector space with that dimension
        rnn_hidden_size = 
        rnn_num_layers = the numbers of the layers in the each LSTM in the model
        z_dim = the encoder encodes the sentence to a z-vector space with that dimension
    ''' 
    def __init__(self, dictionary_size=100, embedding_dim=1100, rnn_hidden_size=600, rnn_num_layers=2, z_dim=1100): #Does embedding_dim should be the same as z_dim?
        super(ParaphraseModel, self).__init__()
        self.embedding = nn.Embedding(dictionary_size, embedding_dim) #should be replaced in word embedding like word2vec
        self.encoder = Encoder(embedding_dim, rnn_hidden_size, rnn_num_layers, z_dim)
        self.decoder = Decoder(embedding_dim, rnn_hidden_size, rnn_num_layers, dictionary_size)
        self.cel = nn.CrossEntropyLoss(ignore_index=-1)  #cross entrpoy
        self.dictionary_size = dictionary_size
        self.embedding_dim = embedding_dim

    def train_model(self, xo, xp, xo_len, xp_len, kld_coef=1):
        logits, z, mu, logvar = self.AE_forward(xo, xp, xo_len, xp_len)

        cel_loss = self.cel(logits.view(-1, self.dictionary_size).contiguous(), xp.cuda().view(-1))
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        total_loss = cel_loss + kld_coef * kl_loss  
        #print(cel_loss, kl_loss)
        return total_loss

    def AE_forward(self, xo, xp, xo_len, xp_len):
        xo_embed = self.embedding(xo.cuda())
        xp_embed = self.embedding(xp.cuda())

        mu, logvar = self.encoder(xo_embed, xp_embed, xo_len, xp_len)
        std = torch.exp(0.5*logvar)
        nd = Normal(torch.ones_like(mu), torch.zeros_like(std))
        z = nd.sample() * std + mu
        logits = self.decoder(xo_embed, z, xo_len, xp_len)
        return logits, z, mu, logvar

    def infer(self, xo, xo_len):
        xo_embed = self.embedding(xo.cuda())
        _, (hT, cT) = self.decoder.ose(xo_embed, xo_len)
        completed_sentences = torch.zeros(len(xo_embed))
        sentences = []
        mu, sigma = torch.zeros(len(xo), self.embedding_dim), torch.ones(len(xo), self.embedding_dim)
        nd = Normal(mu, sigma)
        z = nd.sample().cuda()
        out = hT[-1]
        steps = 0
        while not all(completed_sentences):
            real_inp = torch.cat((z, out), 1).unsqueeze(1)
            output, (hT, cT) = self.decoder.pse(real_inp, torch.tensor([1] * len(z)), h0=hT, c0=cT)
            out = hT[-1]
            probs = self.decoder.linear(out)
            
            topwords = [word_probs.topk(1)[1] for word_probs in probs]
            for j, result in enumerate(topwords):
                if int(result)==EOS_TOKEN:
                    completed_sentences[j] = 1
            
            sentences.append(topwords)
            steps+=1
            if steps == MAX_PARA_LENGTH:
                break
        
        return sentences


       
if __name__ == '__main__':
    a = ParaphraseModel()
    b = torch.LongTensor([[1, 2, 3], [4, 5, 2], [1, 1, 1], [1, 2, 1]])
    c = torch.LongTensor([3, 2, 1, 2])
    print(a.train_model(b, b, c, c))
    res = (a.infer(b,c))

        

            
    
