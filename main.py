# -*- coding: utf-8 -*-
# @Author  : jingyi

import glob
import os
import shutil
import sys
import wandb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
sys.path.append(os.path.join('..', os.path.dirname(__file__)))
#from dataloader import Lidarcapv2_Dataset
from datasets.lidarcap import Lidarcapv2_Dataset as Lidarcapv2_Dataset
# from model.tpointnet2 import TPointNet2
from model.outline import Outline
from loss.losses import Loss
from util.crafter import Trainer
from util.crafter import Evaler
#torch.set_num_threads(1)       # 避免cpu利用率出现不合理的暴增

#torch.backends.`cudnn.benchmark = True

parser = argparse.ArgumentParser()
# bs
parser.add_argument('--bs', type=int, default=8,
                    help='input batch size for training (default: 24)')
parser.add_argument('--eval_bs', type=int, default=8,
                    help='input batch size for evaluation')
# threads
parser.add_argument('--threads', type=int, default=1,
                    help='Number of threads (default: 4)')

# epochs
parser.add_argument('--epochs', type=int, default=10000,
                    help='Traning epochs (default: 100)')

parser.add_argument('--debug', action='store_true', default=False)

parser.add_argument('--eval', action='store_true', default=False)

# parser.add_argument('-g', '--gpus', default=7, type=int)

parser.add_argument('--group_name', default='test', type=str)
parser.add_argument('--wandbid', default=None, type=str)

parser.add_argument('--config', type=str, default='base.yaml')

parser.add_argument('--ckpt_path', type=str, default=None)

args = parser.parse_args()

def main():

    if args.threads == 0:
        torch.set_num_threads(1)  # 此行可以规避这个bug：threads为0时cpu占用出现不合理的暴增
    else:
        torch.set_num_threads(1 + args.threads)

    wandb.config.update(args, allow_val_change=True)

    model_dir = os.path.join(os.path.dirname(__file__), 'output', str(wandb.run.id))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=False)

    for f in glob.glob(os.path.join(os.path.dirname(__file__), "*.py")) + glob.glob(
         os.path.join(os.path.dirname(__file__), f"{args.config}.yaml")):
        shutil.copy(f, model_dir)

    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(open(os.path.join(os.path.dirname(__file__), f'configs/{args.config}')))
    wandb.config.update(cfg, allow_val_change=True)
    config = wandb.config

    ##   dataset processing   ##
    testset = Lidarcapv2_Dataset(cfg.TestDataset)
    testset.__getitem__(0)

    valid_loader = torch.utils.data.DataLoader(
        testset,
        num_workers=config.threads,
        batch_size=config.eval_bs,
        shuffle=False,
        pin_memory=True,
    )

    if not args.eval:
        trainset = Lidarcapv2_Dataset(cfg.TrainDataset)  # 在训练时将use_aug设为False会有更快的收敛速度，而且没有任何性能下降

        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=config.threads,
            batch_size=config.bs,
            shuffle=True,
            pin_memory=True,
        )

        loader = dict(Train=train_loader, Valid=valid_loader)

    else:
        loader = dict(Valid=valid_loader)

    ##  model init  ##
    if cfg.MODEL.TpointNet2:
        net = TPointNet2()

    if cfg.MODEL.Outline:
        net = Outline(cfg)

    net.cuda()
    
    ##  init loss   ##
    loss_func = Loss(cfg)
    loss_func.cuda()

    # train or val
    if not args.eval:
        trainer = Trainer(net, loader, loss_func, config, 0, args.eval)
    else:
        trainer = Evaler(net,loader,loss_func, config,0,args.eval)
    
    start_epoch = 0 
    # if wandb.run.resumed:
    #     a = os.path.dirname(__file__), 'output', args.ckpt_path, 'BEST_time_model.pth'
    #     print(f'Resumed from ckpt_path:{a}')
    #     start_epoch, best_performance = trainer.resume_pretrained(a)
    # elif args.ckpt_path is not None:
    #     a = os.path.dirname(__file__), 'output', args.ckpt_path, 'BEST_time_model.pth'
    #     start_epoch, best_performance = trainer.resume_pretrained(a)
    # else:
    #     start_epoch = 0
    if args.ckpt_path is not None:
        a = os.path.join(os.path.dirname(__file__), 'output', args.ckpt_path, 'best_model.pth')
        start_epoch, best_performance = trainer.resume_pretrained(a)

    trainer.fit(start_epoch, config.epochs, model_dir)


if __name__ == "__main__":

    local = dict(
        WANDB_BASE_URL='your_wandb_link',
        WANDB_ENTITY='your_wandb',
        WANDB_API_KEY='your_wandb'
    )

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    for k in local.keys():
        os.environ[k]=local[k]
    
    if args.debug:
        os.environ["WANDB_MODE"] = 'disabled'
        
    wandb.init(project='v1', entity='your_wandb', id=args.wandbid)
    main()