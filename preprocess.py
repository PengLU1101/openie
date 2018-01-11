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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"\"", r"'", s)
    s = re.sub(r"/", r"", s)
    s = re.sub(r"  ", r" ", s)
    s = re.sub(r"[^a-zA-Z0-9.,'!?$[]\"<>-]+", r" ", s)
    return s

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
def build_word_dict(sentences):
    word_dict = defaultdict(int)
    for s in sentences:
        for fields in s:
            word_dict[fields] += 1
    return word_dict
def build_label_dict(sentences):
    label_dict = defaultdict(int)
    for s in sentences:
        for fields in s:
            label_dict[fields] += 1
    return label_dict
def len_argsort(seq):
    """
    Function using to sort a list which its items are also list,the key to sort it is
     the length of list in it.
    Argument: seq(list of list, e.g. [[1, 2, 3], [4, 5], [c, d, e, g, f, k]])
    Output: the sorted index(e. g [1, 0, 2 ])
    """
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))


def test():
    path = '/home/peng-lu/Projects/deepie/data/extractions-all-labeled.txt'
    pairs = load_data(path)
    pprint.pprint(pairs[0])

if __name__ == '__main__':
    test()
