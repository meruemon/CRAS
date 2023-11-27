import os
import numpy as np
import torch
import json
import argparse
import random
from pathlib import Path


def save_args(args, dst):
    if dst is None:
        return
    with open(dst, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_args(src):
    if src is not None and Path(src).exists():
        with open(src, 'r') as f:
            ns = json.load(f)
        return argparse.Namespace(**ns)


def make_logdir(path, text):
    if os.path.isfile(path):
        pass
    else:
        with open(path, 'w') as f:
            f.write(text)


def save_checkpoint(args, epoch, model1, model2, optimizer1, optimizer2, acc,
                    save_best=False, save_ckpt=False, save_dir=None):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(model1).__name__

    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict1': model1.state_dict(),
        'state_dict2': model2.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'acuracy': acc,
        'args': args
    }
    if save_best:
        best_path = save_dir + '/model_best.pth'
        torch.save(state, best_path)
        print("\nSaving current best: model_best.pth at: \n{} ...".format(
            best_path))
    elif save_ckpt:
        ckpt_path = save_dir + '/model_epoch_{}.pth'.format(epoch)
        torch.save(state, ckpt_path)
        print(
            "\nSaving current model: model_epoch_{}.pth at: \n{} ...".format(
                epoch, ckpt_path)
        )
    else:
        ckpt_path = save_dir + '/model_checkpoint.pth'
        torch.save(state, ckpt_path)


def seed_setting(seed):
    os.environ['PYTHONHASEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def seed_worker(worker_id):
    """
        Dataloader seed worker settings
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def adjust_learningrate(epoch, lr, optimizer1, optimizer2,
                        milestones=50, gamma=10):
    if epoch >= milestones:
        lr /= gamma
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
