import lightning as L
from lightning.pytorch.core import LightningDataModule

from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from typing import List

from src.datamodules.catdog_dataset import CatDogDataset


class CatDogData(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 2,
        splits: List[float] = [0.8,0.1,0.1],
        pin_memory: bool = False  
    ):  
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers 
        self._split = splits
        self._pin_memory = pin_memory
    
    @property 
    def data_dir(self):
        return self._data_dir
    
    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
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
    
    
    def setup(self, stage=None):

        self.full_dataset  =  ImageFolder(self._data_dir)

        labels = np.array([sample[1] for sample in self.full_dataset.samples])

        # Initialize stratified split with 80-10-10 proportions
        sss_train_val = StratifiedShuffleSplit(n_splits=1, train_size = self._split[0] ,test_size=(self._split[1]+self._split[2]))
        train_idx, test_val_idx = next(sss_train_val.split(np.zeros(len(labels)), labels))

        # Further split test and validation from the 20% using another stratified split
        sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=(self._split[1]+self._split[2])/2)
        val_idx, test_idx = next(sss_val_test.split(np.zeros(len(test_val_idx)), labels[test_val_idx]))

        # Assign indices to datasets
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, test_val_idx[val_idx])
        self.test_dataset = Subset(self.full_dataset, test_val_idx[test_idx])

        self.train_dataset = CatDogDataset(self.train_dataset, self.train_transform)
        self.val_dataset = CatDogDataset(self.val_dataset, self.val_transform)
        self.test_dataset = CatDogDataset(self.test_dataset, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=self._pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)