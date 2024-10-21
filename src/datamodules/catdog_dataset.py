from torch.utils.data import Dataset 

class CatDogDataset(Dataset):

    def __init__(self, subset, transforms):
        super().__init__()
        self.subset = subset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        
        img, label = self.subset[index]
        if self.transforms:
            img = self.transforms(img)
        return img, label
