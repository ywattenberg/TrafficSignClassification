from numpy import float32
import pandas as pd
import torch
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset



class SignDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_transform=None):

        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform 

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.label.loc[idx, 'path'])
        image = io.imread(img_name)
        image = float32(np.transpose(image, (2,0,1)))
        label = self.label.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        
        return image, label

        