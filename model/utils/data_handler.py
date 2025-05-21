import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataHandler:
    """This class creates train and testing DataLoaders.
        It takes a tranining directory and a test directory and turns 
        them into PyTorch Datasets and then into PyTorch DataLoaders.
        Adapted from https://www.learnpytorch.io/06_pytorch_transfer_learning/ and 
        https://www.learnpytorch.io/05_pytorch_going_modular/
    """
    def __init__(self, 
                 train_dir: str,
                 val_dir: str, 
                 test_dir: str,
                 num_workers: int = os.cpu_count(),
                 batch_size: int = 32,
                 transform: transforms.Compose = None
                 ):
        """This is a DataHandler class constructor.

        Args:
            train_dir (str): train directory
            test_dir (str): test directory
            num_workers (int, optional): number of workers. 
            Defaults to os.cpu_count().
            batch_size (int, optional): batch size. Defaults to 32.
            transform (transforms.Compose, optional): tranformations applied 
            to images. Defaults to None.
        """
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
        
        self.__class_names = self.__train_data.classes
        
        self.__train_dataloader = self.__create_dataloader(self.__train_data, shuffle=True)
        self.__val_dataloader = self.__create_dataloader(self.__val_data)
        self.__test_dataloader = self.__create_dataloader(self.__test_data)

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
    
    @property
    def train_dataloader(self) -> DataLoader:
        """This is a getter for train dataloader."""
        return self.__train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        """This is a getter for val dataloader."""
        return self.__val_dataloader
    
    @property
    def test_dataloader(self) -> DataLoader:
        """This is a getter for test dataloader."""
        return self.__test_dataloader
    
    @property
    def class_names(self) -> list[str]:
        """This is a getter for class names."""
        return self.__class_names
