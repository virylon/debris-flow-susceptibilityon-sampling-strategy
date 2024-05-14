import os
import logging
import torch
import scipy.io
import numpy as np
from os import listdir
from os.path import splitext
from glob import glob
from torch.utils import data
from torch.utils.data import Dataset
from utils import Normalization
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

factors = ["aspect", "curvature", "dem", "disfault","disriver", "lithology", "pga", "rain", "slope", "twi"]
           

class BasicDataset(Dataset):
    def __init__(self, patch_dir):
        # self.patch_dir = patch_dir
        csv = np.genfromtxt(patch_dir, delimiter=",")
        self.dataX = csv[:, 2:][1:]
        self.dataY = csv[:, 1][1:]

    def __len__(self):
        return len(self.dataX)


    def __getitem__(self, i):

        x = self.dataX[i]
        y = self.dataY[i]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        x = x.unsqueeze(-1)
        x = F.normalize(x, dim=0)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)


        return {"label": y, "data": x}
