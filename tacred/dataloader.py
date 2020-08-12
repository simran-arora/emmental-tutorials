import os
import random
import re
import numpy as np
import json
import torch

from emmental.data import EmmentalDataset

from utils import constant, helper, vocab

#some options in this file: choosing random seed, shuffling the data order, cleanup of the examples

# """
# Data loader for TACRED json files.
# """

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
    print(token_len)
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


class TACREDDataset(EmmentalDataset):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, name, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        with open(filename) as infile:
            data = json.load(infile)

        data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data] 
        raw_labels = [d[-1] for d in data]
        
        #print("LABEL INFO: ", raw_labels[0], self.labels[0]) # 0 no_relation
        self.num_examples = len(data)
        
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)] # split into batches
        raw_labels = [raw_labels[i:i+batch_size] for i in range(0, len(raw_labels), batch_size)]
        
#         print("PRINTING SOME FACTS: ")
#         print(type(data)) = list
#         print(type(data[0])) = list
#         print(type(data[0][0])) = tuple
#         print(len(data)) = 1
#         print(len(data[0])) = batch size
#         print(len(data[0][0])) = num columns


        self.data = data
        X_dict = {"data": []}
        X_dict["data"] = data
        
        Y_dict = {}
        labels_lst = []
        
        for lab in raw_labels:
            labels_lst.append(lab)

        labels_lst[-1] += [0] * (50 - len(labels_lst[-1])) # TODO: resolve this padding cleanly

        #Y_dict["label"] = torch.from_numpy(np.array(labels_lst[0]))
        Y_dict["label"] = torch.from_numpy(np.array(labels_lst))
        #Y_dict["label"] = get_long_tensor(labels_lst,50)#, dtype=torch.int)
               
        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict)

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
#         words_lst = []
#         masks_lst = []
#         pos_lst = []
#         ner_lst = []
#         deprel_lst = []
#         subj_positions_lst = []
#         obj_positions_lst = []
#         relation_lst = []
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
            processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]
            
            
#             words_lst.append(words)
#             masks_lst.append(masks)
#             pos_lst.append(pos)
#             ner_lst.append(ner)
#             deprel_lst.append(deprel)
#             subj_positions_lst.append(subj_positions)
#             obj_positions_lst.append(obj_positions)
#             relation_lst.append(relation)
           
        #return processed
        return processed


    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """

        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
#         print("BATCH INFO: ")
#         print(len(batch))
#         print(type(batch))
#         print()
        assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        
        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)
        obj_positions = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])

        # print(self.X_dict.keys())
        self.X_dict['data'][key] = (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels) # orig_idx
        self.Y_dict['label'][key] = rels
        x_dict = {name: feature[key] for name, feature in self.X_dict.items()}
        y_dict = {name: label[key] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    