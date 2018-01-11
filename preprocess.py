import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext


import numpy as np
import os
import sys
import re
import unicodedata
import pprint
from collections import defaultdict

def load_data(path):
    with open(path, "r", encoding='iso-8859-15') as f:
        pairs = []
        pair = []
        for line in f.readlines():
            if not line[0].isdigit():
                if pair:
                    pairs.append(pair)
                    pair = []
                else:
                    pair.append(line[:-1])
            else:
                string = filter_long50(line[1:-1])
                if string:
                    pair.append(string)
        return pairs
def filter_long50(string):
    count = 0
    for i in string.split('\t'):
        count += len(i.split(" "))
    if count <= 50:
        return string



def test():
    path = '/home/peng-lu/Projects/deepie/data/extractions-all-labeled.txt'
    pairs = load_data(path)
    pprint.pprint(pairs[0])

if __name__ == '__main__':
    test()
