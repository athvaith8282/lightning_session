import pytest
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

import hydra

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hydra import initialize, compose

@pytest.fixture
def train_cfg() -> DictConfig:
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train"
            )
        return cfg

@pytest.fixture
def test_cfg() -> DictConfig:
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="eval"
            )
        return cfg

@pytest.fixture
def infer_cfg() -> DictConfig:
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="infer"
            )
        return cfg

@pytest.fixture
def datamodule(train_cfg):
    return hydra.utils.instantiate(train_cfg.data)
