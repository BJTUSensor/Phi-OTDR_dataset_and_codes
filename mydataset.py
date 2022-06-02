import numpy as np
import os
import scipy.io as scio
from torch.utils.data import Dataset

def normalize(data):                           # 归一化到0-255
    rawdata_max = max(map(max, data))
    rawdata_min = min(map(min, data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = round(((255 - 0) * (data[i][j] - rawdata_min) / (rawdata_max - rawdata_min)) + 0)
    return data

class MyDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []
        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_path = self.root_dir + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(data_path):
            print(data_path + 'does not exist!')
            return None
        rawdata = scio.loadmat(data_path)['data']  # 10000,12 uint16
        rawdata = rawdata.astype(int)       # int32
        data = normalize(rawdata)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
