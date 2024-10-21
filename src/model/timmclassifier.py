from typing import Any
from lightning.pytorch.core import LightningModule

from torch import optim 
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
from timm import create_model
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss 

    def validation_step(self, batch, batch_idx):
        x,y = batch 
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/acc", self.val_accuracy, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.test_accuracy(preds, y)
        self.confusion_matrix(preds.argmax(dim=1), y)  # Update confusion matrix
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.test_accuracy, prog_bar=True)

    def on_test_epoch_end(self):
        cm = self.confusion_matrix.compute()  # Compute confusion matrix
        
        # Save confusion matrix as an image
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Class {i}' for i in range(self.hparams.num_classes)],
                    yticklabels=[f'Class {i}' for i in range(self.hparams.num_classes)])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')  # Save the figure
        plt.close()  # Close the figure to free memory
        
        self.confusion_matrix.reset()  # Reset for next test

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
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }
