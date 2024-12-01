import os
import math
import errno
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def mkdir_p(path):

    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def adjust_learning_rate(optimizer, train_configuration, epoch, training_epoch_num, args):

    cos_inner = np.pi * (epoch % training_epoch_num)
    cos_inner /= (training_epoch_num)
    cos_out = np.cos(cos_inner) + 1
    learning_rate = float(train_configuration['learning_rate'] / 2 * cos_out)

    optimizer.param_groups[0]['lr'] =  learning_rate / 10
    optimizer.param_groups[1]['lr'] =  learning_rate / 10
    optimizer.param_groups[2]['lr'] =  learning_rate
    optimizer.param_groups[3]['lr'] =  learning_rate
    optimizer.param_groups[4]['lr'] =  learning_rate
    optimizer.param_groups[5]['lr'] =  learning_rate

    for param_group in optimizer.param_groups:
        print(param_group['lr'])


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = checkpoint + '/' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, checkpoint + '/model_best.pth.tar')