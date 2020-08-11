import argparse
import logging
import sys

import torch
import emmental
from data import create_dataloaders, load_data
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
    
    # TODO: this is from TACRED code -- is this already handled somewhere in emmental?
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(1234)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Log configuration into files
#     cmd_msg = " ".join(sys.argv)
#     logger.info(f"COMMAND: {cmd_msg}")
#     write_to_file(f"{Meta.log_path}/cmd.txt", cmd_msg)

#     logger.info(f"Config: {Meta.config}")
#     write_to_file(f"{Meta.log_path}/config.txt", Meta.config)

    ### DATASETS ###
    datasets = {}
    data = []  

    print(args.task) ## TODO: figure out why this converts to 'T' 'A' .. 'D' and then get rid of task_names
    task_names = ['TACRED']
    for task_name in task_names:
        for split in ["train", "dev", "test"]:
            filename = args.data_dir + '/' + split + '.json'
            data, labels = load_data(
                filename,
                opt, 
                vocab, 
                evaluation=False
            )
            
            X_dict = {
                "data": data
            }
            Y_dict = {
                "labels": labels
            }

            if task_name not in datasets:
                datasets[task_name] = {}

            datasets[task_name][split] =EmmentalDataset(
                    name=task_name,
                    X_dict=X_dict,
                    Y_dict=Y_dict
            )
            datasets[task_name]["nclasses"] = opt['num_class']
    print('done with datasets')

    ### DATA LOADERS ###
    dataloaders = []
    for task_name in task_names:
        for split in ["train", "dev", "test"]:
            dataloaders.append(
                EmmentalDataLoader(
                    task_to_label_dict={task_name: "labels"},
                    dataset=datasets[task_name][split],
                    split=split,
                    batch_size=args.batch_size,
                    shuffle=True if split == "train" else False,
                )
            )
            logger.info(f"Built dataloader for {task_name} {split} set.") 
            print('done with dataloaders: ', split)

    tasks = {
        task_name: create_task(
            task_name, args, opt, emb_matrix # TODO: was task_name, args, datasets[task_name]["nclasses"], emb_matrix.
        )
        for task_name in task_names
    }
    print('Made tasks!')

    
    ### MODEL ###
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
#     logger.info(f"Metrics: {scores}")
#     write_to_json_file(f"{Meta.log_path}/metrics.txt", scores)

    
    
#     ### CHECKPOINTING ###
#     if args.checkpointing:
#         logger.info(
#             f"Best metrics: "
#             f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
#         )
#         write_to_file(
#             f"{Meta.log_path}/best_metrics.txt",
#             emmental_learner.logging_manager.checkpointer.best_metric_dict,
#         )
        
# #     model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
# #     model.save(model_file, epoch)
# #     if epoch == 1 or dev_f1 > max(dev_f1_history):
# #         copyfile(model_file, model_save_dir + '/best_model.pt')
# #         print("new best model saved.")
# #     if epoch % opt['save_epoch'] != 0:
# #         os.remove(model_file)
        # model
    
    
    
    
#     model = RelationModel(opt, emb_matrix=emb_matrix)

#     id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
#     dev_f1_history = []
#     current_lr = opt['lr']

#     global_step = 0
#     global_start_time = time.time()
#     format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
#     max_steps = len(train_batch) * opt['num_epoch']

#     # start training
#     for epoch in range(1, opt['num_epoch']+1):
#         train_loss = 0
#         for i, batch in enumerate(train_batch):
#             start_time = time.time()
#             global_step += 1
#             loss = model.update(batch)
#             train_loss += loss
#             if global_step % opt['log_step'] == 0:
#                 duration = time.time() - start_time
#                 print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
#                     opt['num_epoch'], loss, duration, current_lr))

#         # eval on dev
#         print("Evaluating on dev set...")
#         predictions = []
#         dev_loss = 0
#         for i, batch in enumerate(dev_batch):
#             preds, _, loss = model.predict(batch)
#             predictions += preds
#             dev_loss += loss
#         predictions = [id2label[p] for p in predictions]
#         dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
    
#         train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
#         dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
#         print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
#                 train_loss, dev_loss, dev_f1))
#         file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

#         # save
#         model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
#         model.save(model_file, epoch)
#         if epoch == 1 or dev_f1 > max(dev_f1_history):
#             copyfile(model_file, model_save_dir + '/best_model.pt')
#             print("new best model saved.")
#         if epoch % opt['save_epoch'] != 0:
#             os.remove(model_file)
    
#         # lr schedule
#         if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
#                 opt['optim'] in ['sgd', 'adagrad']:
#             current_lr *= opt['lr_decay']
#             model.update_lr(current_lr)

#         dev_f1_history += [dev_f1]
#         print("")

