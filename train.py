import torch
import os
from torchvision import transforms
import wandb
import time

from argparse import ArgumentParser
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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

parser = ArgumentParser(description='InceptionV1 monosemantic feature learning')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use')
parser.add_argument('--model_to_hook', default='inceptionv1', type=str, help='model to hook')
parser.add_argument('--layer_name', default='mixed4a', type=str, help='layer to hook')
parser.add_argument('--patch_size', default=None, type=maybe_None_or_int, help='patch size')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--channels', default = 508, type=int, help='number of channels')
parser.add_argument('--hidden_size', default = 10 * 508, type=int, help='hidden size')
parser.add_argument('--l1_coeff', default = 1e-2, type=float, help='l1 coefficient')
parser.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
parser.add_argument('--seed', default = 0, type=int, help='seed')
parser.add_argument('--max_epochs', default = 100, type=int, help='max epochs')
parser.add_argument('--gpus', default = 1, type=int, help='number of gpus')
parser.add_argument('--optimizer', default = 'Adam', type=str, help='optimizer')
parser.add_argument('--num_workers', default = 8, type=int, help='number of workers')
parser.add_argument('--resume_from_checkpoint', default = None, type=str, help='resume from checkpoint')
parser.add_argument('--save_top_k', default = 1, type=int, help='save top k')
parser.add_argument('--save_path', default = '/mnt/home/dheurtel/ceph/02_checkpoints/', type=str, help='save path')
parser.add_argument('--wandb_project', default = 'monosemantic_dictionnary_learning', type=str, help='wandb project')
parser.add_argument('--model_name', default = 'inceptionv1_mixed4a_monosemantic', type=str, help='model name')
parser.add_argument('--logger', default = 'wandb', type=str, help='logger')


args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.save_path, exist_ok=True)

CONFIG = vars(args)

if args.dataset == 'imagenet':
    ## 1st tries to make a /tmp/imagenet/ folder.
    ## If it exists, waits for it to be complete (another job on the node is downloading it)

    ## Waits for a random amount of time to avoid all jobs downloading at the same time
    ## Should use ddp with rank 0... but right now jobs are not launched with ddp/submitted individually
    print('Waiting for a random amount of time to avoid all jobs downloading at the same time')
    time.sleep(120* torch.rand(1).item())
    direxists = os.path.exists('/tmp/blocking')

    if not direxists:
        os.makedirs('/tmp/blocking') ## Will block other jobs from downloading imagenet on the same node
        print('Downloading imagenet from ceph and unzipping it')
        os.system('bash dataload.sh')
        print('Done')
    else:
        print('Waiting for imagenet to be downloaded')
        while not os.path.exists('/tmp/imagenet/val'):
            time.sleep(120)
            print('Waiting for imagenet to be downloaded for 2 minutes')
        n_classes_val = len(os.listdir('/tmp/imagenet/val'))
        while n_classes_val < 1000:
            time.sleep(120)
            n_classes_val = len(os.listdir('/tmp/imagenet/val'))
            print('Waiting for imagenet to be downloaded for 2 minutes')
        print('Imagenet downloaded by another job')
    
    # dirnotempy = os.path.exists('/tmp/imagenet/train')
    # if dirnotempy:
    #     l = os.listdir('/tmp/imagenet/train')
    #     if len(l)==0:
    #         dirnotempy = False
    # if not dirnotempy:
    #     print('Downloading imagenet from ceph and unzipping it')
    #     os.system('bash dataload.sh')
    #     print('Done')

    print('Building datasets and dataloaders')
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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Done')
else: 
    raise NotImplementedError



dict_learner = DictionnaryLearner(args.hidden_size, 
                                  args.channels, 
                                  args.l1_coeff, 
                                  args.seed, 
                                  model_to_hook=args.model_to_hook, 
                                  layer_to_hook=args.layer_name, 
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
                                  wandb_project=args.wandb_project)

logger_type = args.logger

if logger_type == 'wandb':
    logger = pl.loggers.WandbLogger(project=args.wandb_project, name=args.model_name)
    logger.watch(dict_learner)
    logger.log_hyperparams(CONFIG)
    default_root_dir = os.path.join(args.save_path)
elif logger_type == 'tensorboard':
    logger = pl.loggers.TensorBoardLogger(args.save_path, name=args.model_name)
    default_root_dir = None
elif logger_type == 'None':
    logger = None
    default_root_dir = os.path.join(args.save_path, args.model_name)

trainer = pl.Trainer(max_epochs = args.max_epochs, logger=logger, default_root_dir=default_root_dir)#devices=args.gpus
trainer.fit(dict_learner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if logger_type == 'wandb':
    wandb.finish()