import lightning as L
from lightning.pytorch.loggers import Logger
import hydra

from omegaconf import DictConfig
from typing import List 
from loguru import logger


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        logger.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers