import random

import h5py
import numpy as np
from torch.utils.data import Dataset

from utils import ToTensor


class TrainingDataset(Dataset):
    def __init__(self, file_path, frame_size, frame_shift, nsamples=16000 * 2):
        with open(file_path, 'r') as train_file_list:
            self.file_list = [line.strip() for line in train_file_list.readlines()]
        self.nsamples = nsamples
        self.to_tensor = ToTensor
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]
        reader.close()
        size = feature.shape[0]
        if size >= self.nsamples:
            start = random.randint(0, max(0, size - self.nsamples))
            noisy = feature[start:start + self.nsamples]
            clean = label[start:start + self.nsamples]
        else:
            units = self.nsamples // size
            clean = []
            noisy = []
            for i in range(units):
                clean.append(label)
                noisy.append(feature)
            clean.append(label[: self.nsamples % size])
            noisy.append(feature[: self.nsamples % size])
            clean = np.concatenate(clean, axis=-1)
            noisy = np.concatenate(noisy, axis=-1)
        noisy = self.to_tensor(noisy)  # [sig_len,]
        clean = self.to_tensor(clean)  # [sig_len,]
        return noisy, clean


class EvalDataset(Dataset):
    def __init__(self, file_path, frame_size=400, frame_shift=100):
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.to_tensor = ToTensor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        noisy = reader['noisy_raw'][:]
        clean = reader['clean_raw'][:]
        reader.close()
        clean = self.to_tensor(clean)
        noisy = self.to_tensor(noisy)
        # print(noisy.shape, clean.shape)
        return noisy, clean
