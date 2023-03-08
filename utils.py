import os
import random

import numpy as np
import torch

from argparse import ArgumentParser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item() * 100
    else:
        return (pred == label).type(torch.FloatTensor).mean().item() * 100


def is_debug():
    import sys
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


def update_ema_variables(ema_model, model, alpha):
    # Use the true average until the exponential average is more correct

    ema_dict = ema_model.state_dict()
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        ema_dict[k] = alpha * ema_dict[k] + (1 - alpha) * v
    ema_model.load_state_dict(ema_dict)


def add_arguments(parser: ArgumentParser):
    # add PROGRAM level args
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--model', type=str, default='ce')
    parser.add_argument('--network', type=str, default='resnet18')
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_epochs', type=int, default=500)

    parser.add_argument('--val_every_n_epoch', type=int, default=1)

    # add optimizer args
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--point', type=int, nargs='+', default=(100, 50, 20))
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    parser.add_argument('--warmup', action='store_true', default=False)

    # add dataset args
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--train_transform', type=str, default='simclr')

    return parser


def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # os.environ['MASTER_PORT'] = '10129'
    print('using gpu:', args.gpus)
    gpus = range(len(args.gpus.split(',')))
    args.gpus = ','.join(str(g) for g in gpus)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def model_load_weights(init_weights, model):
    model_state_dict = model.state_dict()
    try:
        pretrained_dict = torch.load(init_weights)
    except:
        import pickle

        with open(init_weights, 'rb') as fp:
            pretrained_dict = pickle.load(fp)
    keys = ['params', 'state_dict']
    for k in keys:
        if k in pretrained_dict:
            pretrained_dict = pretrained_dict[k]
            break
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    print(pretrained_dict.keys())
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
