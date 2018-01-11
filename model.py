import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn import utils as nn_utils
#import utils
import math
from pprint import pprint as pp

class discriminator(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 dropout_p,
                 ):
        super(discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout_p
        #self.cell_type = cell_type
        self.embedding = nn.Embedding(input_size, 
                                      hidden_size,
                                      padding_idx=0)
        self.rnn_sents = nn.GRU(hidden_size, 
                                hidden_size, 
                                n_layers, 
                                dropout_p)
        self.rnn_triple = nn.GRU(hidden_size, 
                                hidden_size, 
                                n_layers, 
                                dropout_p)
        self.fc_sent = nn.Linear(hidden_size,
                                 hidden_size)
        self.fc_triple = nn.Linear(hidden_size,
                                 hidden_size)
        self.compare_fc = nn.Linear(hidden_size,
                                    hidden_size,
                                    bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_sent,
                input_triple,
                mask_sent,
                mask_triple):
        embedded_sent = self.embedding(input_sent).transpose(1, 0)
        embedded_triple = self.embedding(input_triple).transpose(1, 0)

        out_sent, hid_sent = self.rnn_sents(embedded_sent)
        out_triple, hid_triple = self.rnn_triple(embedded_triple)

        out_sent = out_sent * mask_sent.unsqueeze(-1)
        out_triple = out_triple * mask_triple.unsqueeze(-1)

        sent_vec = self.fc_sent(hid_sent[-1])
        triple_vec = self.fc_triple(hid_triple[-1])
        print(triple_vec.size())
        score = torch.bmm(sent_vec.unsqueeze(1), 
                          self.compare_fc(triple_vec).unsqueeze(-1))

        prob = self.sigmoid(score.squeeze(-1).squeeze(-1))
        return prob

def test():
    sent = (torch.randn(5, 10))**2
    sent = Variable(sent.long())

    triple = (torch.randn(5, 3))**2
    triple = Variable(triple.long())

    model = discriminator(100, 20, 3, 0.2)
    x = model(sent, triple)

    print(x)

if __name__ == '__main__':
    test()




