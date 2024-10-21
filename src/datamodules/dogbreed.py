import gdown
import zipfile
import os 

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from lightning.pytorch.core.datamodule import LightningDataModule 

from src.datamodules.dogbreed_dataset import DogBreedDataset

from typing import List


class DogBreedDataModule(LightningDataModule):
    def __init__(self, gdrive_url: str, data_dir: str = "./data",download_dir: str = "./data", batch_size: int = 64, num_workers: int = 2, splits: List[float] = [0.8, 0.1, 0.1], pin_memory: bool = False):
        super().__init__()
        self._gdrive_url = gdrive_url
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._splits = splits
        self._pin_memory = pin_memory
        self._download_dir = download_dir
    
    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])
    
    @property
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform,
        ])
    
    @property 
    def test_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform,
        ])
    
    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def data_dir(self):
        return self._data_dir
    
    def prepare_data(self):
        
        gdrive_url = self._gdrive_url
        if not os.path.exists(self._download_dir):
            os.makedirs(self._download_dir)
        output = os.path.join(self._download_dir, "dogbreed.zip")
        gdown.download(gdrive_url, output, quiet=False)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(os.path.join(self.data_dir, "dogbreed"))
    
    def setup(self, stage: str = "train"):
        full_dataset = ImageFolder(root=self.data_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, self._splits)
        self.train_dataset = DogBreedDataset(self.train_dataset, self.train_transform)
        self.val_dataset  = DogBreedDataset(self.val_dataset, self.val_transform)
        self.test_dataset = DogBreedDataset(self.test_dataset, self.test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=self._pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)