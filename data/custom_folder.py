import os
from torchvision import datasets

class CustomImageFolder(datasets.ImageFolder):
    """
    ImageFolder
    """
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path
