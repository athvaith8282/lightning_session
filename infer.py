import lightning as L

from model.dogbreed_model import DogBreedClassifier
from datamodules.dogbreed import DogBreedDataModule

from lightning.pytorch.loggers import TensorBoardLogger

from torchvision.datasets import ImageFolder

import torch 
from torch.nn import functional as F
import random 

from torchvision import transforms
from PIL import Image as PILImage

from torchmetrics import Accuracy

def get_dataset(root_dir):

    return ImageFolder(root=root_dir)

def get_n_image_paths(dataset, n = 10):

   samples = random.sample(dataset.samples, n)
   return samples

if __name__ == "__main__": 

    best_model_path = "logs/test_1/version_2/checkpoints/epoch=2-step=39.ckpt"

    img_datadir = "./data/dogbreed/dataset"

    dataset = get_dataset(img_datadir)

    samples = get_n_image_paths(dataset, 10)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    paths, labels = zip(*samples)
    batch = torch.stack([transform(PILImage.open(path).convert('RGB')) for path in paths])

    model = DogBreedClassifier.load_from_checkpoint(best_model_path) 
    model.eval()
    batch = batch.to(model.device)
    labels = torch.tensor(labels).to(model.device)
    inference_accuracy = Accuracy(task="multiclass", num_classes=10).to(model.device)


    with torch.no_grad():
        outputs = model(batch)
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.to(model.device)
        inference_accuracy(probabilities, labels)

    print(f"Inference Accuracy: {inference_accuracy.compute()}")

