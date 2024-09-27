from typing import Any
from lightning.pytorch.core import LightningModule

from torch import optim 
 
import torch.nn.functional as F
from torchmetrics import Accuracy
from timm import create_model


class DogBreedClassifier(LightningModule):

    def __init__(self, lr:float = 1e-3):
        
        super().__init__()
        self.lr = lr 

        self.model = create_model(
            model_name="resnet18",
            pretrained=True,
            num_classes = 10 
            )
        
        self.train_accuracy = Accuracy('multiclass', num_classes=10)
        self.val_accuracy = Accuracy('multiclass', num_classes=10)
        self.test_accuracy = Accuracy('multiclass', num_classes=10)

        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        x,y = batch 
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x,y = batch 
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.test_accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), self.lr)