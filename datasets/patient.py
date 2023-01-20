import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
import os

class MIMIC(Dataset):
    def __init__(self, data_dir, time_len=1024, train=True, normalize=True):
        ''' data shape: (time_len, 1)'''
        self.data_dir = data_dir
        files = [i for i in os.listdir(data_dir) if '.mat' == i[-4:]]

        self.data_continuous = []
        for filename in files:
            mat = sio.loadmat(os.path.join(data_dir, filename))['val']

            pleth = mat[0]
            abp = mat[1]

            pleth = pleth[~np.isnan(pleth)]
            abp = abp[~np.isnan(abp)]

            if normalize:
                pleth = (pleth - pleth.min()) / (pleth.max() - pleth.min()) * 2 - 1
                abp = (abp - abp.min()) / (abp.max() - abp.min()) * 2 - 1

            file_data = np.stack([pleth, abp], axis=-1)

            file_data_split = np.split(file_data, np.arange(time_len,len(file_data),time_len), axis=0)[:-1]

            self.data_continuous.extend(file_data_split)

        self.data_continuous = np.stack(self.data_continuous, axis=0).astype(np.float32, copy=False)

        if train:
            self.data = self.data_continuous[:int(len(self.data_continuous)*0.8)]
        else:
            self.data = self.data_continuous[int(len(self.data_continuous) * 0.8):]
        self.channel_dims = 2
        self.feat_dims = time_len


    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

