from torch.utils.data import Dataset

class CatRabbitDataset(Dataset):

    def __init__(self, dataset, transforms):

        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
    
    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        if self.transforms:
            image = self.transforms(image)
        return image,label

    def __len__(self):

        return len(self.dataset)