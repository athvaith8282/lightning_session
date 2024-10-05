from typing import Any
from lightning.pytorch.core import LightningModule

from torch import optim 
 
import torch.nn.functional as F
from torchmetrics import Accuracy
from timm import create_model


class TimmClassifier(LightningModule):

    def __init__(self, 
        base_model_name: str, 
        num_classes: int, 
        pretrained: bool = True, 
        lr:float = 1e-3, 
        weight_decay: float = 1e-5,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6):
        
        super().__init__()
        self.save_hyperparameters()


        self.model = create_model(
            model_name=self.hparams.base_model_name,
            pretrained=self.hparams.pretrained,
            num_classes = self.hparams.num_classes 
            )
        
        self.train_accuracy = Accuracy('multiclass', num_classes=self.hparams.num_classes)
        self.val_accuracy = Accuracy('multiclass', num_classes=self.hparams.num_classes)  
        self.test_accuracy = Accuracy('multiclass', num_classes=self.hparams.num_classes)

    
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

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }