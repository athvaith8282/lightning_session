import lightning as L
from lightning.pytorch.core import LightningDataModule
from torchvision import transforms

from torchvision.datasets import ImageFolder

from torch.utils.data import  Subset,DataLoader

from sklearn.model_selection import StratifiedShuffleSplit


from typing import List

import numpy as np

from src.datamodules.catrabbit_dataset import CatRabbitDataset

class CatRabbit(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 2,
        splits: List[float] = [0.8,0.2],
        pin_memory: bool = False, 
        resize: tuple = (224,224)
    ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers 
        self._split = splits
        self._pin_memory = pin_memory
        self.validated = False
        self.resize = resize
    
    @property
    def data_dir(self):
        return self._data_dir

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
         return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def  test_transform(self):
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            self.normalize_transform,
        ])
    
    def setup(self, stage: str = "fit") -> None:

        if stage == "fit":

            self.dataset = ImageFolder(f"{self._data_dir}/train-cat-rabbit")

            targets = self.dataset.targets

            sss_train_val = StratifiedShuffleSplit(1, test_size=self._split[1], train_size=self._split[0])

            train_idx , val_idx = next(sss_train_val.split(np.zeros(len(targets)), targets))

            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)

            self.train_dataset = CatRabbitDataset(self.train_dataset, self.train_transform)
            self.val_dataset = CatRabbitDataset(self.val_dataset, self.val_transform)
        
        elif stage == "test":

            self.test_dataset = ImageFolder(f"{self._data_dir}/val-cat-rabbit")
            self.test_dataset = CatRabbitDataset(self.test_dataset, transforms=self.test_transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self._batch_size,
            shuffle = True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self._batch_size,
            shuffle = True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory
            )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self._batch_size,
            shuffle = True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory
            )