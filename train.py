from model.dogbreed_model import DogBreedClassifier
from datamodules.dogbreed import DogBreedDataModule
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger

L.seed_everything(seed=42, workers=True)

if __name__ == "__main__":

    model = DogBreedClassifier()
    checkpoint_callbacks = ModelCheckpoint(monitor="val_loss")

    data_module = DogBreedDataModule()

    trainer = L.Trainer( 
        log_every_n_steps=1,
        accelerator="auto",
        callbacks=[
            checkpoint_callbacks
        ],
        max_epochs=3,
        logger = TensorBoardLogger(save_dir="logs", name="test_1")
    )

    trainer.fit(model, data_module)
    print(checkpoint_callbacks.best_model_path)