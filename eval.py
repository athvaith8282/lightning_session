import lightning as L

from model.dogbreed_model import DogBreedClassifier
from datamodules.dogbreed import DogBreedDataModule

from lightning.pytorch.loggers import TensorBoardLogger


if __name__ == "__main__": 

    best_model_path = "logs/test_1/version_2/checkpoints/epoch=2-step=39.ckpt"

    model = DogBreedClassifier.load_from_checkpoint(best_model_path) 

    data_module = DogBreedDataModule()

    trainer = L.Trainer(
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="logs", name="test_1", version="version_2")
        )

    trainer.test(model, data_module)
