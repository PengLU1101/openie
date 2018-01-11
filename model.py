import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn import utils as nn_utils
import utils
import math
from pprint import pprint as pp

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.n_layers = num_layers
        self.dropout = dropout_p
        self.cell_type = cell_type
        self.ngpu = n_gpu
        self.main = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True)
        nn.ConstantPad2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf*4)
        )

    def forward(self):
        pass
class discriminator(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
