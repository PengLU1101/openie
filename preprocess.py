import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext
from torch.utils.data import Dataset, DataLoader


import numpy as np
import os
import sys
import re
import unicodedata
import pprint
from collections import defaultdict
from random import shuffle

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
        triples = []
        labels = []
        triple = ""
        sentence = ""
        sentences = []
        for line in f.readlines():
            if line[0].isdigit():
                triple, label = split(line)
                if triple and label:
                    triples.append(triple)
                    labels.append(label)
                    sentences.append(normalizeString(sentence))
                else:
                    pass
            else:
                sentence = line[: -1]
        return sentences, triples, labels
def split(line):
    triple, label = '', ''
    fields = line[2:-1].split('\t')
    for i in fields[:-1]:
        triple += i[1: -1] + " "
    label = fields[-1]
    return normalizeString(triple), label

def build_word_dict(sentences):
    word_dict = defaultdict(int)
    for s in sentences:
        for fields in s.split(' '):
            word_dict[fields] += 1
    return word_dict
def build_label_dict(sentences):
    label_dict = defaultdict(int)
    for s in sentences:
        for fields in s:
            label_dict[fields] += 1
    return label_dict
def create_mapping_with_unk(dico):
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_word = {index + 2: w[0] for (index, w) in enumerate(sorted_items)}
    word_to_id = {v: k for k, v in id_to_word.items()}
    #sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    #word_to_vector = [ for i in id_to_word.keys() ]

    id_to_word[0] = "<pad>"
    word_to_id["<pad>"] = 0
    id_to_word[1] = "<unk>"
    word_to_id["<unk>"] = 1
    return word_to_id, id_to_word

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    id_to_item[0] = "O"
    item_to_id["O"] = 0
    return item_to_id, id_to_item
def len_argsort(seq):
    """
    Function using to sort a list which its items are also list,the key to sort it is
     the length of list in it.
    Argument: seq(list of list, e.g. [[1, 2, 3], [4, 5], [c, d, e, g, f, k]])
    Output: the sorted index(e. g [1, 0, 2 ])
    """
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

def filter_long_index(triples, lim):
    index = [len(triple) < lim for triple in triples]
    return index

def random_array(sent, triple, label, sword_to_id, label_to_id):
    sents = np.array([[sword_to_id[i] for i in s.split(" ")] for s in sent])
    triples = np.array([[sword_to_id[i] for i in s.split(" ")] for s in triple])
    labels = np.array([[label_to_id[i] for i in s] for s in label])
    #index = filter_long_index(triples, 30)
    '''
    sent = np.array(sents)
    triple = np.array(triples)
    label = np.array(labels)
    '''
    idx = np.random.permutation(sent.shape[0])
    return sents[idx], triples[idx], labels[idx]

def write_txt(path, file):
    with open(path, "a") as f:
        for i in file:
            f.write(i + '\n')
class sent_triple_dataset(Dataset):
    #self.sent_triple = a
    pass
def test():
    pwd = os.getcwd()
    path = os.path.join(pwd, "data/extractions-all-labeled.txt")
    #path = '/home/peng-lu/Projects/deepie/data/extractions-all-labeled.txt'
    sent, triple, label= load_data(path)
    sent_path = os.path.join(pwd, "data/sents.txt")
    triple_path = os.path.join(pwd, "data/triples.txt")
    label_path = os.path.join(pwd, "data/labels.txt")

    word_dict = build_word_dict(sent + triple)
    label_dict = build_label_dict(label)
    print(len(word_dict))


    sword_to_id, id_to_sword = create_mapping_with_unk(word_dict)
    label_to_id, id_to_label = create_mapping(label_dict)
    triples = np.array([[sword_to_id[i] for i in s.split(" ")] for s in triple])
    labels = np.array([[label_to_id[i] for i in s] for s in label])
    index = filter_long_index(triples, 20)
    idx = np.array(index)
    ll = labels[idx]
    print("sum", sum(index))
    print("sum l", sum(labels))
    print("sum ll", sum(ll))
    files = np.loadtxt(path)
    print(files[:10])



if __name__ == '__main__':
    test()
