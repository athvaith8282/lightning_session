import lightning as L
import hydra

from omegaconf import DictConfig
from loguru import logger
from pathlib import Path

from src.utils.logging_utils import task_wrapper, setup_logger
from src.utils.instantiate_utils import instantiate_loggers

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@task_wrapper
def test(
    cfg: DictConfig,
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

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):


    setup_logger(log_file= Path(cfg.paths.output_dir)/"test.log")

    data_module = hydra.utils.instantiate(cfg.data)

    model = hydra.utils.instantiate(cfg.test_model)

    loggers = instantiate_loggers(cfg.loggers)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers)

    test(cfg, trainer, model, data_module)


if __name__ == "__main__": 


    main()

    
