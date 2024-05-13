import torch
import os
import shutil
import numpy as np


class Checkpoint(object):
    def __init__(self, start_epoch=None, best_val_loss=float("inf"),
                 state_dict=None, optimizer=None):
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.state_dict = state_dict
        self.optimizer = optimizer

    def save(self, is_best, filename, best_model):
        print(f'Saving checkpoint at {filename}')
        torch.save(self, filename)
        if is_best:
            print(f'Saving the best model at {best_model}')
            shutil.copyfile(filename, best_model)

    def load(self, filename):
        if os.path.isfile(filename):
            print(f'Loading checkpoint from {filename}\n')
            checkpoint = torch.load(filename, map_location='cpu')
            self.start_epoch = checkpoint.start_epoch
            self.best_val_loss = checkpoint.best_val_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
        else:
            raise ValueError(f'No checkpoint found at {filename}')
