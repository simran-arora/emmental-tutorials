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

        data, words_lst, masks_lst, pos_lst, ner_lst, deprel_lst, subj_positions_lst, obj_positions_lst, relation_lst = self.preprocess(data, vocab, opt)

        # shuffle for training TODO: add this back in 
#         if not evaluation:
#             indices = list(range(len(data)))
#             random.shuffle(indices)
#             data = [data[i] for i in indices]

        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data] 
        raw_labels = [d[-1] for d in data]
        
        #print("LABEL INFO: ", raw_labels[0], self.labels[0]) # 0 no_relation
        self.num_examples = len(data)

        self.data = data
        X_dict = {}
        X_dict["words"] = words_lst
        X_dict["masks"] = masks_lst
        X_dict["pos"] = pos_lst
        X_dict["ner"] = ner_lst
        X_dict["deprel"] = deprel_lst
        X_dict["subj"] = subj_positions_lst
        X_dict["obj"] = obj_positions_lst
        X_dict["rels"] = relation_lst
        
        Y_dict = {}
        Y_dict["label"] = torch.from_numpy(np.array(relation_lst)) # TODO: get code review
               
        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict)

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        words_lst = []
        masks_lst = []
        pos_lst = []
        ner_lst = []
        deprel_lst = []
        subj_positions_lst = []
        obj_positions_lst = []
        relation_lst = []
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
            
            # NOTE: Emmental should take care of padding
            words = torch.tensor(tokens)
            words_lst.append(words)
            masks = torch.eq(words, 0)
            masks_lst.append(torch.tensor(masks))
            pos_lst.append(torch.tensor(pos))
            ner_lst.append(torch.tensor(ner))
            deprel_lst.append(torch.tensor(deprel))
            subj_positions_lst.append(torch.tensor(subj_positions))
            obj_positions_lst.append(torch.tensor(obj_positions))
            relation_lst.append(torch.tensor(relation))
           
        #return processed
        return processed, words_lst, masks_lst, pos_lst, ner_lst, deprel_lst, subj_positions_lst, obj_positions_lst, relation_lst


    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """

        x_dict = {name: feature[key] for name, feature in self.X_dict.items()}
        y_dict = {name: label[key] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    