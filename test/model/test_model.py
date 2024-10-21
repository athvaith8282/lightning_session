import pytest
import torch
import hydra
from omegaconf import DictConfig

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.timmclassifier import TimmClassifier

@pytest.fixture
def model(train_cfg: DictConfig) -> TimmClassifier:
    return hydra.utils.instantiate(train_cfg.model)

def test_timmclassifier_init(model: TimmClassifier):
    assert isinstance(model, TimmClassifier)
    assert model.model is not None
    assert model.train_accuracy is not None
    assert model.val_accuracy is not None
    assert model.test_accuracy is not None

def test_timmclassifier_forward(model: TimmClassifier):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, model.hparams.num_classes)

def test_timmclassifier_training_step(model: TimmClassifier):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, model.hparams.num_classes, (batch_size,))
    loss = model.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

def test_timmclassifier_validation_step(model: TimmClassifier):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, model.hparams.num_classes, (batch_size,))
    model.validation_step((x, y), 0)
    # Check if metrics are logged (you might need to mock the logger)

def test_timmclassifier_test_step(model: TimmClassifier):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, model.hparams.num_classes, (batch_size,))
    model.test_step((x, y), 0)
    # Check if metrics are logged (you might need to mock the logger)

def test_timmclassifier_configure_optimizers(model: TimmClassifier):
    optimizer_config = model.configure_optimizers()
    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config
    assert isinstance(optimizer_config["optimizer"], torch.optim.Optimizer)
    assert isinstance(optimizer_config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)