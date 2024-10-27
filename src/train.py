import lightning as L
from lightning.pytorch.loggers import Logger

import hydra
from omegaconf import DictConfig

from typing import List 
from loguru import logger
from pathlib import Path

from src.utils.logging_utils import setup_logger,task_wrapper

from src.utils.instantiate_utils import instantiate_callbacks, instantiate_loggers

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os 
from dotenv import load_dotenv, set_key

def set_seed(seed: int = 42):
    L.seed_everything(seed=42, workers=True)

@task_wrapper
def train(
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    logger.info("Starting training!")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    logger.info("saving best model to .env model_path")
    set_key('.env', 'CHECKPOINT_PATH', trainer.checkpoint_callback.best_model_path)
    logger.info(f"Training metrics:\n{train_metrics}")

    return train_metrics

@task_wrapper
def test(
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    logger.info("Starting testing!")
    if trainer.checkpoint_callback.best_model_path:
        logger.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        logger.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    logger.info(f"Test metrics:\n{test_metrics}")

    return test_metrics[0] if test_metrics else {}

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):

    set_seed()
    setup_logger(log_file=Path(cfg.paths.output_dir)/"train.log")

    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    callbacks = instantiate_callbacks(cfg.callbacks)
    loggers = instantiate_loggers(cfg.loggers)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks= callbacks,
        logger = loggers
    )

    train_metrics = {}
    if cfg.get("train"):
        train_metrics = train(trainer, model, data_module)

    test_metrics = {}
    if cfg.get("test"):
        test_metrics = test(trainer, model, data_module)

    all_metrics = {**train_metrics, **test_metrics}

    optimization_metric = all_metrics.get(cfg.get("optimization_metric"))
    if optimization_metric is None:
        logger.warning(f"Optimization metric '{cfg.get('optimization_metric')}' not found in metrics. Returning 0.")
        return 0.0
    
    return optimization_metric

if __name__ == "__main__":

    try:
        load_dotenv()
        main()
    except Exception as e:
        logger.exception(e)