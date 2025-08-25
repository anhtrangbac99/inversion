"""Some parts based on https://github.com/yang-song/score_sde_pytorch"""

from torch.utils.data import DataLoader, Dataset
import numpy as np
from mpi4py import MPI
import blobfile as bf
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, datasets
import torch
from PIL import Image
import os

class UniformDequantize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return (torch.rand(sample.shape) + sample*255.)/256.


def get_dataset(config, uniform_dequantization=False, train_batch_size=None,
                eval_batch_size=None, num_workers=8):
    train_data = PairedImageDataset(config)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    return train_loader

class PairedImageDataset(Dataset):
    def __init__(self, config):
        super(PairedImageDataset, self).__init__()
        self.config = config

        self.paths = os.listdir(self.config.data.blur_paths)
        self.blur_paths = [os.path.join(self.config.data.blur_paths,i) for i in self.paths]
        self.sharp_paths = [os.path.join(self.config.data.sharp_paths,i) for i in self.paths]
        
        self.preprocessor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.config.data.image_size,self.config.data.image_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    def __getitem__(self, index):
        blur,sharp = Image.open(self.blur_paths[index],mode='r'),Image.open(self.sharp_paths[index],mode='r')
        blur,sharp = self.preprocessor(blur),self.preprocessor(sharp)
        # image = im.permute(2, 0, 1) 
        return blur,sharp
        

    def __len__(self):
        return len(self.paths)