import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger

from torchvision.datasets import ImageFolder

import torch 
from torch.nn import functional as F
import random 

from torchvision import transforms
from PIL import Image as PILImage

from torchmetrics import Accuracy

import hydra
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig

from src.utils.logging_utils import setup_logger

import rootutils
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def get_dataset(root_dir):

    return ImageFolder(root=root_dir)

def get_n_image_paths(dataset, n = 10):

   samples = random.sample(dataset.samples, n)
   return samples


@hydra.main(version_base=None, config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    # Replace checkpoint_path in the Hydra config with the value from the .env file
    if 'CHECKPOINT_PATH' in os.environ:
        cfg.infer_model.checkpoint_path = os.environ['CHECKPOINT_PATH']
    
    logger.info(f"Using model from path {cfg.infer_model.checkpoint_path}")

    setup_logger(Path(cfg.paths.output_dir)/"infer.log")

    img_datadir = cfg.paths.data_dir

    dataset = get_dataset(img_datadir)

    samples = get_n_image_paths(dataset, cfg.infer_n_images)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    paths, labels = zip(*samples)
    batch = torch.stack([transform(PILImage.open(path).convert('RGB')) for path in paths])

    model = hydra.utils.instantiate(cfg.infer_model)
    model.eval()
    batch = batch.to(model.device)
    labels = torch.tensor(labels).to(model.device)
    inference_accuracy = Accuracy(task="multiclass", num_classes=10).to(model.device)


    with torch.no_grad():
        outputs = model(batch)
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.to(model.device)
        inference_accuracy(probabilities, labels)

    logger.info(f"Inference Accuracy: {inference_accuracy.compute()}")



if __name__ == "__main__": 

    main()

    