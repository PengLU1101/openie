import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import random
import time
import math
import numpy as np
from random import shuffle
from pprint import pprint as pp
import argparse
import json
import faiss

import model
import preprocess
import utils
import dataloader
import load_kb
USE_CUDA = torch.cuda.is_available()


parser = argparse.ArgumentParser(description='deepie')
parser.add_argument('--data', type=str,
    default='/u/lupeng/Project/code/vqvae_kb/data/processed/',help='path to the data')
parser.add_argument('--rnn_cell', type=str, default='LSTM',
    help='type of recurrent net (rnn, LSTM, GRU)')

parser.add_argument('--optim', type=str, default='RMSprop',
    help='type of optim (Adam, RMS)')

parser.add_argument('--para_init', type=float, default=0.001,
    help='initial model parameters')
parser.add_argument('--emsize', type=int, default=50,
    help='size of word embeddings')
parser.add_argument('--concept_size', type=int, default=50,
    help='size of concept embeddings')
parser.add_argument('--nhid', type=int, default=50,
    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
    help='number of layers')
parser.add_argument('--num_kb', type=int, default=1,
    help='number of kb')
parser.add_argument('--lr', type=float, default=0.01,
    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=80,
    help='upper epoch limit')
parser.add_argument('--patience', type=int, default=20,
    help='waiting epoch numbers')
parser.add_argument('--batch_size', type=int, default=64,
    metavar='N', help='batch size')
parser.add_argument('--max_len', type=int, default=120,
    help='max sequence length')
parser.add_argument('--L2', type=float, default=0.0,
    help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5,
    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1000,
    help='random seed')
parser.add_argument('--print_every', type=int, default=1,
    metavar='N', help='report interval for how many epochs')
parser.add_argument('--print_every_train', type=int, default=40,
    metavar='N', help='report interval for how many training')
parser.add_argument('--model_path', type=str,
    default='./models',
    help='path to save the final model')
parser.add_argument('--mode', type=str, default='train',
    help='type of mode (train, inference)')
parser.add_argument('--pretrain_em', type=int, default=1,
    help='use pre trained embedding')
args = parser.parse_args()

num_epoch = args.epochs
patience = args.patience
print_every = args.print_every
print_every_train = args.print_every_train
emb_size = args.emsize
concept_size = args.concept_size
hidden_size = args.nhid
num_kb = args.num_kb
num_layer = args.nlayers
lr = args.lr
L2 = args.L2
dropout_p = args.dropout
batch_size = args.batch_size
mode = args.mode
pretrain_em = args.pretrain_em
model_path = args.model_path

############################### Load data ###############################
cwd = os.getcwd()
datapth = os.path.join(cwd, "extractions-all-labeled.txt")
sent, triple, label= load_data(path)
print(len(sent))
sent_path = os.path.join(pwd, "data/sents.txt")
triple_path = os.path.join(pwd, "data/triples.txt")
label_path = os.path.join(pwd, "data/labels.txt")

word_dict = preprocess.build_word_dict(sent+triple)
label_dict = preprocess.build_label_dict(label)

sword_to_id, id_to_sword = preprocess.create_mapping_with_unk(word_dict)
label_to_id, id_to_label = preprocess.create_mapping(label_dict)

sents, triples, labels = preprocess.random_array(sent,
                                                 triple,
                                                 label,
                                                 sword_to_id,
                                                 label_to_id)


print("Load data...")
sents_train = sents[:5798]
sents_dev = sents[5798:]
triples_train = triples[:5798]
triples_dev = triples[5798:]
labels_train = labels[:5798]
labels_dev = labels[5798:]



print("#### Data detail ####")
print("The number of data in train set %d" % len(sents_train))
print("The number of data in dev set %d" % len(sents_dev))
print("The number of tokens in traina and dev set: %d" %len(sword2id))


#########################################################################

############################## Build model ##############################
discriminator = model.discriminator(input_size=len(sword2id),
                                    em_size=emb_size,
                                    concept_size=concept_size,
                                    dropout=dropout_p)

######
if pretrain_em:
    discriminator.init_para(pre_weights)
if USE_CUDA:
    discriminator = discriminator.cuda()

optimizer = getattr(optim, args.optim)(discriminator.parameters(), weight_decay=L2)

criterion = nn.NLLLoss()

print("#####Model detail#####")
print(discriminator)
if USE_CUDA:
    print("USE_CUDA")

#########################################################################

def prepare_data(src_seqs,
                 tgt_seqs,
                 volatile=False):
    length_src = [len(s) for s in src_seqs]
    length_tgt = [len(s) for s in tgt_seqs]
    #assert length_src == length_tgt

    n_samples = len(src_seqs)
    maxlen_src = max(length_src)
    maxlen_tgt = max(length_tgt)
    assert maxlen_src == maxlen_tgt

    src = torch.LongTensor(n_samples, maxlen_src)
    src[:n_samples, :maxlen_src] = 0
    mask = torch.zeros(n_samples, maxlen_src)
    for idx, s in enumerate(src_seqs):
        src[idx, :length_src[idx]] = torch.LongTensor(s)
        mask[idx, :length_src[idx]] = 1
    tgt = torch.LongTensor(n_samples, maxlen_tgt)
    tgt[:n_samples, :maxlen_tgt] = 0
    for idx, s in enumerate(tgt_seqs):
        tgt[idx, :length_tgt[idx]] = torch.LongTensor(s)
    src = Variable(src, volatile=volatile)
    tgt = Variable(tgt, volatile=volatile)
    mask_src = Variable(mask, requires_grad=False)
    if USE_CUDA:
        src = src.cuda()
        tgt = tgt.cuda()
        mask_src = mask_src.cuda()
    return src, tgt, mask_src

def batchize(src_list,
             tgt_list,
             batch_size,
             volatile=False):
    datapairslist = []
    len_src = len(src_list)
    ################################

    for batch_index in range(0, len_src, batch_size):
        batch_len = min(batch_size, len_src - batch_index)
        src_seqs = src_list[batch_index: batch_index+batch_len]
        tgt_seqs = tgt_list[batch_index: batch_index+batch_len]

        if src_seqs != []:
            datapairs = prepare_data(src_seqs,
                                     tgt_seqs,
                                     volatile)
            datapairslist.append(datapairs)

    return datapairslist

len_trainset = len(inputseqs_train)

def inference(discriminator, dev_src, dev_tgt):
    discriminator.eval()
    datapairslist = batchize(dev_src,
                             dev_tgt,
                             batch_size,
                             volatile=True)
    score = 0
    len_datalist = len(datapairslist)
    prec, rec, f1 = 0, 0, 0
    _, e_v = load_kb.loadvec()
    flat_config = faiss.GpuIndexIVFFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFFlat(res, args.concept_size, 1000, faiss.METRIC_L2, flat_config)
    index.train(e_v)
    index.add(e_v)
    for i, item in enumerate(datapairslist):
        src_seqs, tgt_seqs, mask = item
        batch_len, maxlen = src_seqs.size()

        ###
        embbed, pre_kb_emb = encoder(src_seqs)
        embbed = embbed * mask.unsqueeze(-1)
        pre_kb_emb = (pre_kb_emb * mask.unsqueeze(-1)).permute(1, 0, 2)
        if not isinstance(pre_kb_emb, np.ndarray):
            pre_kb_emb = pre_kb_emb.data.cpu().numpy()
        v_list = []
        for item in pre_kb_emb:
            D, I = index.search(item, args.num_kb)
            v_can = e_v[I]
            v_list.append(torch.from_numpy(v_can))
        v = Variable(torch.stack(v_list, 0))
        if USE_CUDA:
            v = v.cuda()
        v = v * mask.transpose(1, 0).unsqueeze(-1).unsqueeze(-1)
        ###
        scores, preds = vqcrf.inference(embbed, v, mask)
        micro_prec, micro_rec, micro_f1 = evaluate_acc(tgt_seqs, preds)
        prec += micro_prec
        rec += micro_rec
        f1 += micro_f1

    return prec / len_datalist, rec / len_datalist, f1 / len_datalist



def train_epoch(discriminator, train_src, train_tgt, epoch_index, lr):
    discriminator.train()
    datapairslist = batchize(train_src,
                             train_tgt,
                             batch_size)

    epoch_loss = 0
    start_time = time.time()
    #encoder_optimizer = getattr(optim, args.optim)(encoder.parameters(), weight_decay=L2)
    #vqcrf_optimizer = getattr(optim, args.optim)(vqcrf.parameters(), weight_decay=L2)

    len_traintensorlist = len(train_src)
    idx_list = list(range(len_traintensorlist))
    shuffle(datapairslist)

    _, e_v = load_kb.loadvec()
    flat_config = faiss.GpuIndexIVFFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFFlat(res, args.concept_size, 1000, faiss.METRIC_L2, flat_config)
    index.train(e_v)
    index.add(e_v)
    for i, item in enumerate(datapairslist):
        total_loss = 0
        src_seqs, tgt_seqs, mask = item
        batch_len, maxlen = src_seqs.size()
        encoder.zero_grad()
        vqcrf.zero_grad()

        embbed, pre_kb_emb = encoder(src_seqs)
        embbed = embbed * mask.unsqueeze(-1)
        pre_kb_emb = (pre_kb_emb * mask.unsqueeze(-1)).permute(1, 0, 2)
        if not isinstance(pre_kb_emb, np.ndarray):
            pre_kb_emb = pre_kb_emb.data.cpu().numpy()
        v_list = []
        for item in pre_kb_emb:
            D, I = index.search(item, args.num_kb)
            v_can = e_v[I]
            v_list.append(torch.from_numpy(v_can))
        v = Variable(torch.stack(v_list, 0))
        if USE_CUDA:
            v = v.cuda()
        v = v * mask.transpose(1, 0).unsqueeze(-1).unsqueeze(-1)

        neglogscore = vqcrf(embbed, v, tgt_seqs, mask).mean()
        #print("neglogscore", neglogscore.size())
        #decoder_hidden = decoder.init_hidden(batch_len)
        neglogscore.backward()
        torch.nn.utils.clip_grad_norm(vqcrf.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)

        encoder_optimizer.step()
        vqcrf_optimizer.step()
        epoch_loss += neglogscore.data[0]
        print_loss = neglogscore.data[0] / len(tgt_seqs)


        if (i % print_every_train == 0 and i != 0) or (len_traintensorlist-1 == i):
            using_time = time.time() - start_time
            print('| epoch %3d | %4d/%5d batches | ms/batch %5.5f | '
                'loss %5.15f | ppl: %5.2f |}' %(
                    epoch_index, i, len_trainset // batch_size,
                    using_time * 1000 / print_every_train, print_loss,
                    math.exp(print_loss)))
            print_loss = 0
            start_time = time.time()


    epoch_loss = epoch_loss/len_trainset
    return epoch_loss

def train(discriminator,
          train_src,
          train_tgt,
          num_epoch,
          print_every,
          lr,
          L2,
          patience):
    """
    datapairslist = batchize(inputseqs_dev,
                             realtags_dev,
                             batch_size=16,
                             volatile=True)
    """

    try:
        best_val_loss = 0
        best_epoch = 0
        print_loss_total = 0
        start = time.time()
        print("Begin training...")

        for epoch_index in range(1, num_epoch+1):
            train_loss = train_epoch(encoder, vqcrf, train_src, train_tgt, epoch_index, lr)
            prec, rec, val_loss = inference(encoder, vqcrf, inputseqs_dev, realtags_dev)

            print_loss_total += train_loss
            if not best_val_loss or val_loss > best_val_loss:

                if not os.path.isfile(args.model_path):
                    model_path = os.path.join(args.model_path, 'discriminator.pkl')
                    torch.save(discriminator.state_dict(), model_path)
                    best_val_loss = val_loss
                    best_epoch = epoch_index
            else:
                lr /= 2
                if epoch_index - best_epoch > 5:
                    optimizer = getattr(optim, args.optim)(discriminator.parameters(), weight_decay=L2)

                if epoch_index - best_epoch > patience:
                    print("Stop early at the %d epoch, the best epoch: %d"
                        %(epoch_index, best_epoch))
                    break
            if epoch_index % print_every == 0:
                print_loss_total = 0
                print_loss_avg = print_loss_total / min(print_every, num_epoch-epoch_index)

                print('-' * 100)
                print('%s (%d %d%%)| best_f1:%.4f| F1:%.10f| prec:%.4f| '
                    'rec:%.4f| best_epoch:%d'
                    % (utils.timeSince(start, epoch_index / (num_epoch+1)),
                    epoch_index, epoch_index / (num_epoch+1) * 100,
                    best_val_loss, val_loss, prec, rec, best_epoch))
                print('-' * 100)

    except KeyboardInterrupt:
        print('-' * 100)
        print('Stop early... the best epoch is %d' %best_epoch)


if __name__ == '__main__':
    if mode == "train":
        train(discriminator, inputseqs_train, realtags_train, num_epoch, print_every, lr, L2, patience)
    else:
        inference(discriminator, inputseqs_dev, realtags_dev)
