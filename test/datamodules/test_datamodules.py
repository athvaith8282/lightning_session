import pytest
import hydra
import os

from pathlib import Path

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def test_dogbreed_datamodule_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Update this assertion to check for the correct total size
    total_size = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
    assert total_size == sum(len(files) for root, dirs, files in os.walk(datamodule.data_dir) if any(file.endswith('.jpg') for file in files))


def test_dogbreed_datamodule_train_val_test_splits(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    # Check if the splits are correct (80% train, 10% val, 10% test)
    total_train_val_test = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
    assert len(datamodule.train_dataset) / total_train_val_test == pytest.approx(
        0.8, abs=0.01
    )
    assert len(datamodule.val_dataset) / total_train_val_test == pytest.approx(0.1, abs=0.01)

    # Check if test dataset is separate
    assert len(datamodule.test_dataset) / total_train_val_test == pytest.approx(0.1, abs=0.01)



def test_dogbreed_datamodule_dataloaders(datamodule, train_cfg):
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check if the batch sizes are correct
    assert train_loader.batch_size == train_cfg.data.batch_size
    assert val_loader.batch_size == train_cfg.data.batch_size
    assert test_loader.batch_size == train_cfg.data.batch_size


def test_dogbreed_datamodule_transforms(datamodule):
    assert datamodule.train_transform is not None
    assert datamodule.val_transform is not None
    assert datamodule.test_transform is not None

    # Check if the image size in transforms matches the specified size
    assert datamodule.train_transform.transforms[0].size == (224, 224)
    assert datamodule.val_transform.transforms[0].size == (224, 224)
    assert datamodule.test_transform.transforms[0].size == (224, 224)

def test_dogbreed_datamodule_data_path(datamodule):
    assert Path(datamodule.data_dir).exists()