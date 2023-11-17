import torch
import os
import json
import zipfile
import gzip
import matplotlib.pyplot as plt
from PIL import Image
import torchvision 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import wandb
import tqdm
import torch.nn.functional as F

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
#from zipfile import ZipFile

from zippedimagefolder import ZippedDatasetFolder
import zippedimagefolder

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import pytorch_lightning as pl

from model import DictionnaryLearner

def maybe_None_or_int(arg):
    """Check if arg is None or an int and return the arg
    """
    if arg is None:
        return arg
    elif isinstance(arg, int):
        return arg
    else:
        raise ValueError('arg must be a string or an int')

args = argparse.ArgumentParser(description='InceptionV1 monosemantic feature learning')
args.add_argument('--dataset', default='imagenet', type=str, help='dataset to use')
args.add_argument('--model_to_hook', default='inceptionv1', type=str, help='model to hook')
args.add_argument('--layer_name', default='mixed4a', type=str, help='layer to hook')
args.add_argument('--patch_size', default=None, type=maybe_None_or_int, help='patch size')
args.add_argument('--batch_size', default=512, type=int, help='batch size')
args.add_argument('--channels', default = 508, type=int, help='number of channels')
args.add_argument('--hidden_size', default = 10 * 508, type=int, help='hidden size')
args.add_argument('--l1_coeff', default = 1e-2, type=float, help='l1 coefficient')
args.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
args.add_argument('--seed', default = 0, type=int, help='seed')
args.add_argument('--max_epochs', default = 100, type=int, help='max epochs')
args.add_argument('--gpus', default = 1, type=int, help='number of gpus')
args.add_argument('--optimizer', default = 'Adam', type=str, help='optimizer')
args.add_argument('--num_workers', default = 8, type=int, help='number of workers')
args.add_argument('--resume_from_checkpoint', default = None, type=str, help='resume from checkpoint')
args.add_argument('--save_top_k', default = 1, type=int, help='save top k')
args.add_argument('--save_path', default = '/mnt/home/dheurtel/ceph/02_checkpoints/monosemantic/', type=str, help='save path')
args.add_argument('--wandb_project', default = 'monosemantic_dictionnary_learning', type=str, help='wandb project')
args.add_argument('--model_name', default = 'inceptionv1_mixed4a_monosemantic', type=str, help='model name')
args.add_argument('--logger', default = 'wandb', type=str, help='logger')


args.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.save_path, exist_ok=True)

CONFIG = vars(args)

if args.dataset == 'imagenet':
    dirnotempy = os.path.exists('/tmp/imagenet/train')
    if dirnotempy:
        l = os.listdir('/tmp/imagenet/train')
        if len(l)==0:
            dirnotempy = False
    if not dirnotempy:
        os.system('bash dataload.sh')
    TRAIN_DIR = '/tmp/imagenet/train/'
    TEST_DIR = '/tmp/imagenet/train/'

    traindir = os.path.join('/tmp/imagenet/', 'train')
    valdir = os.path.join('/tmp/imagenet/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    def unormalize(batch):
        ## batch of shape (batch_size, 3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        return batch * std + mean

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))    
else: 
    raise NotImplementedError

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

dict_learner = DictionnaryLearner(args.hidden_size, 
                                  args.channels, 
                                  args.l1_coeff, 
                                  args.seed, 
                                  model_to_hook=args.model_to_hook, 
                                  layer_name=args.layer_name, 
                                  patch_size=args.patch_size, 
                                  lr=args.lr,
                                  optimizer=args.optimizer,
                                  save_path=args.save_path,
                                  model_name=args.model_name,
                                  max_epochs=args.max_epochs,
                                  gpus=args.gpus,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  resume_from_checkpoint=args.resume_from_checkpoint,
                                  save_top_k=args.save_top_k,
                                  wandb_project=args.wandb_project)

logger_type = args.logger

if logger_type == 'wandb':
    logger = pl.loggers.WandbLogger(project=args.wandb_project, name=args.model_name)
    logger.watch(dict_learner)
    logger.log_hyperparams(CONFIG)
    default_root_dir = None
elif logger_type == 'tensorboard':
    logger = pl.loggers.TensorBoardLogger(args.save_path, name=args.model_name)
    default_root_dir = None
elif logger_type == 'None':
    logger = None
    default_root_dir = args.save_path


trainer = pl.Trainer(gpus=args.gpus,max_epoch = args.max_epochs, logger=logger, default_root_dir=default_root_dir, resume_from_checkpoint=args.resume_from_checkpoint, save_top_k=args.save_top_k)
trainer.fit(dict_learner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if logger_type == 'wandb':
    wandb.finish()