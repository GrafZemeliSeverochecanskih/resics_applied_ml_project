from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
class DataHandler:
    def __init__(self, 
                 train_dir, 
                 val_dir,
                 test_dir,
                 num_workers,
                 batch_size,
                 transform = None
                 ):
        self.__train_dir = train_dir
        self.__val_dir = val_dir
        self.__test_dir = test_dir
        self.__num_workers = num_workers
        self.__batch_size = batch_size

        if transform is None:
            self.__transform = self.__create_default_transform()

        self.__train_data = datasets.ImageFolder(self.__train_dir, transform = self.__transform)
        self.__val_data = datasets.ImageFolder(self.__val_dir, transform=self.__transform)
        self.__test_data = datasets.ImageFolder(self.__test_dir, transform=self.__transform)
        
        
        self.__train_dataloader = self.__create_dataloader(self.__train_data, shuffle=True)
        self.__val_dataloader = self.__create_dataloader(self.__val_data)
        self.__test_dataloader = self.__create_dataloader(self.__test_data)

    def __create_dataloader(self, data: datasets, shuffle = False) -> DataLoader:
        """This function creates data loaders."""
        dataloader = DataLoader(
            data,
            batch_size = self.__batch_size,
            shuffle = shuffle,
            num_workers=self.__num_workers,
            pin_memory=True
        )
        return dataloader
    
    def __create_default_transform(self) -> transforms.Compose:
        """This function created default transform.
        Returns:
            transforms.Compose: basic transofrmations applied to images.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
        ])

        return transform
    
    @property
    def train_dataloader(self):
        return self.__train_dataloader

    @property
    def val_dataloader(self):
        return self.__val_dataloader
    
    @property
    def test_dataloader(self):
        return self.__test_dataloader
    
    @property
    def classes(self): 
        return self.__train_data.classes
    
    @property
    def class_to_idx(self): 
        return self.__train_data.class_to_idx