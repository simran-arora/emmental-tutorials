import os
import random
import re
import numpy as np
import json
import random
import torch
import numpy as np
from utils import constant, helper, vocab

#some options in this file: choosing random seed, shuffling the data order, cleanup of the examples


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


"""
Data loader for TACRED json files.
"""
def load_tacred(filename, opt, vocab, evaluation=False):
    with open(filename) as infile:
        data = json.load(infile)
        
    data = preprocess(data, vocab, opt)
    
    # shuffle for training
    if not evaluation:
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()]) ## LABEL_TO_ID should eventually make it into the config file
    #labels = [id2label[d[-1]] for d in data]
    labels = [d[-1] for d in data]
    datas = [d[0:7] for d in data]
    print(labels[0])
    num_examples = len(data)

    # chunk into batches
#     data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
#     data = data
#     print("{} batches created for {}".format(len(data), filename))
    return datas, labels

def preprocess(data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            relation = constant.LABEL_TO_ID[d['relation']]
            
            # TODO: will this work to fix 'ValueError: Label labels should be torch.Tensor, not <class 'list'>.'
            # TODO: make 'word_dropout' optional -- see original code
            words = word_dropout(tokens, opt['word_dropout'])
            words = torch.LongTensor(words)
            masks = torch.eq(words, 0)
            pos = torch.LongTensor(pos)
            ner = torch.LongTensor(ner)
            deprel = torch.LongTensor(deprel)
            subj_positions = torch.LongTensor(subj_positions)
            obj_positions = torch.LongTensor(obj_positions)
            #relation = torch.LongTensor(relation)
            
            processed += [(words, masks, pos, ner, deprel, subj_positions, obj_positions, relation)]
           
        #print(processed[0])
        return processed

# def gold(self):
#    """ Return gold labels as a list. """
#    return self.labels

# def __getitem__(self, key): TODO: how do I implement this in the Emmental Dataloader?
#         """ Get a batch with index. """
#         if not isinstance(key, int):
#             raise TypeError
#         if key < 0 or key >= len(self.data):
#             raise IndexError
#         batch = self.data[key]
#         batch_size = len(batch)
#         batch = list(zip(*batch))
#         assert len(batch) == 7

#         # sort all fields by lens for easy RNN operations
#         lens = [len(x) for x in batch[0]]
#         batch, orig_idx = sort_all(batch, lens)
        
#         # word dropout
#         if not self.eval:
#             words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
#         else:
#             words = batch[0]

#         # convert to tensors TODO: 
#         words = get_long_tensor(words, batch_size)
#         masks = torch.eq(words, 0)
#         pos = get_long_tensor(batch[1], batch_size)
#         ner = get_long_tensor(batch[2], batch_size)
#         deprel = get_long_tensor(batch[3], batch_size)
#         subj_positions = get_long_tensor(batch[4], batch_size)
#         obj_positions = get_long_tensor(batch[5], batch_size)

#         rels = torch.LongTensor(batch[6])

#         return (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]




