import argparse
import logging
import sys

import torch
import emmental
from data import create_dataloaders, load_data
from dataloader import TACREDDataset
from emmental import Meta
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import nullable_string, str2bool, str2list
from modules import EmbeddingLayer
from task import create_task

# tacred-relation repo imports
import os
from datetime import datetime
import time
import numpy as np
import random

from shutil import copyfile
import torch.nn as nn
import torch.optim as optim
from utils import scorer, constant, helper
from utils.vocab import Vocab

logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")
    
    parser = argparse.ArgumentParser()
    application_config.add_argument('--task', type=str, default='')
    application_config.add_argument('--data_dir', type=str, default='dataset/tacred')
    application_config.add_argument('--vocab_dir', type=str, default='dataset/vocab')
    application_config.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    application_config.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
    application_config.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
    application_config.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
    application_config.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
    application_config.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
    application_config.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
    application_config.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
    application_config.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    application_config.add_argument('--no-lower', dest='lower', action='store_false')
    application_config.set_defaults(lower=False)

    application_config.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
    application_config.add_argument('--no-attn', dest='attn', action='store_false')
    application_config.set_defaults(attn=True)
    application_config.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
    application_config.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

    application_config.add_argument('--lr_decay', type=float, default=0.9)
    application_config.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
    application_config.add_argument('--num_epoch', type=int, default=30)
    application_config.add_argument('--batch_size', type=int, default=50)
    application_config.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    application_config.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    application_config.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')

    application_config.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
    application_config.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    application_config.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    application_config.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
    
    application_config.add_argument(
        "--model",
        type=str,
        default="PositionAwareRNN",
        choices=["PositionAwareRNN"], # TODO: add more model options here
        help="Which model to use",
    )

    application_config.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    application_config.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Text Classification Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)
    args = parser.parse_args()
    config = parse_args_to_config(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(1234)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    # make opt
    opt = vars(args)
    opt['num_class'] = len(constant.LABEL_TO_ID)

    # load vocab
    vocab_file = opt['vocab_dir'] + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    opt['vocab_size'] = vocab.size
    emb_file = opt['vocab_dir'] + '/embedding.npy'
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    assert emb_matrix.shape[1] == opt['emb_dim']

    emmental.init(config["meta_config"]["log_path"], config=config)

    # Log configuration into files
#     cmd_msg = " ".join(sys.argv)
#     logger.info(f"COMMAND: {cmd_msg}")
#     write_to_file(f"{Meta.log_path}/cmd.txt", cmd_msg)

#     logger.info(f"Config: {Meta.config}")
#     write_to_file(f"{Meta.log_path}/config.txt", Meta.config)

    ### DATASETS ###
    datasets = {}
    data = []  
    dataloaders = []
    name = 'tacred'
    task_names = [name]
    task_to_label_dict = {task_name: task_name for task_name in task_names}
    
    for split in ["train", "dev", "test"]:
        filename = args.data_dir + '/' + split + '.json'
        dataset = TACREDDataset(
            name,
            filename,
            args.batch_size, 
            opt, 
            vocab, 
            evaluation=False
        )
        logger.info(
            f"Loaded {split} for {name} containing {len(dataset)} samples."
        )
        
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={name: "label"},
                dataset=dataset,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=2,#args.batch_size, #1
                #num_workers=8,
            )
        )
        logger.info(f"Built dataloader for {dataset.name} {split} set.")
        print(f"Built dataloader for {dataset.name} {split} set.")

    tasks = {
        task_name: create_task(
            task_name, args, opt, emb_matrix # TODO: was task_name, args, datasets[task_name]["nclasses"], emb_matrix.
        )
        for task_name in task_names
    }
    print('Made tasks!')

    
#     ### MODEL ###
    model = EmmentalModel(name="Tacred_task")
    print('Made model!')

    if Meta.config["model_config"]["model_path"]:
        model.load(Meta.config["model_config"]["model_path"])
    else:
        for task_name, task in tasks.items():
            model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, dataloaders)


    ### SCORER ###
    scores = model.score(dataloaders)
    print("SCORES: ", scores)
    logger.info(f"Metrics: {scores}")
    #write_to_json_file(f"{Meta.log_path}/metrics.txt", scores)

    
    
    ### CHECKPOINTING ###
    if args.checkpointing:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            f"{Meta.log_path}/best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )
        
# #     model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
# #     model.save(model_file, epoch)
# #     if epoch == 1 or dev_f1 > max(dev_f1_history):
# #         copyfile(model_file, model_save_dir + '/best_model.pt')
# #         print("new best model saved.")
# #     if epoch % opt['save_epoch'] != 0:
# #         os.remove(model_file)
#         model
