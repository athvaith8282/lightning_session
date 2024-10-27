import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger

from torchvision.datasets import ImageFolder

import torch 
from torch.nn import functional as F
import random 

from torchvision import transforms
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageFont


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

    img_datadir = cfg.data.data_dir

    dataset = get_dataset(img_datadir)

    samples = get_n_image_paths(dataset, cfg.infer_n_images)

    transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    paths, labels = zip(*samples)
    batch = torch.stack([transform(PILImage.open(path).convert('RGB')) for path in paths])

    model = hydra.utils.instantiate(cfg.infer_model)
    model.eval()
    batch = batch.to(model.device)
    labels = torch.tensor(labels).to(model.device)

    # Use num_classes from the Hydra config
    inference_accuracy = Accuracy(task="multiclass", num_classes=cfg.num_classes).to(model.device)

    # Create predictions directory if it doesn't exist
    predictions_dir = Path(cfg.paths.pred_dir) 
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        outputs = model(batch)
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.to(model.device)
        inference_accuracy(probabilities, labels)

        # Save predictions with class labels
        for i, (path, prob) in enumerate(zip(paths, probabilities)):
            predicted_class_index = prob.argmax().item()
            predicted_class_label = cfg.classes[predicted_class_index]  # Get class label from config

            # Load the original image
            original_image = PILImage.open(path).convert('RGB')

            # Draw the class label on the image
            draw = ImageDraw.Draw(original_image)

            # Load a font
            font_size = 40  # Adjust font size as needed
            font = ImageFont.load_default(font_size)  # Ensure the font file is available
           
            # Draw the text in the center
            draw.text((10,10), predicted_class_label, fill="red", font=font)  # Adjust color as needed

            # Save the modified image in the predictions folder
            original_image.save(predictions_dir / f"predicted_{i}.png")

    logger.info(f"Inference Accuracy: {inference_accuracy.compute()}")



if __name__ == "__main__": 

    main()

    
