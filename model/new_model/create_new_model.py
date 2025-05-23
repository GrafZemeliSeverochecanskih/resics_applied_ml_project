from pathlib import Path
from utils.data_handler import DataHandler
from new_model.resnet50_custom_model import ResNet50CustomModel
from new_model.trainer import Trainer
from utils.mc_dropout import MonteCarloDropout
from utils.plot_creator import PlotCreator
import os
import torch

class CreateNewModel:
    """This class interconnects different blocks n order to create and
    train new model from scratch.
    """
    def __init__(self,
                 image_path: str,
                 output_dir: str,
                 filename: str = "resnet_50_custom_model_weights",
                 train: str = "train/",
                 val: str = "val/",
                 test: str = "test/",
                 epochs: int = 10,
                 unfreeze_classifier=True,
                 unfreeze_specific_blocks=None
                 ):
        """This is CreateNewModel class constructor."""
        self.__image_path = Path(image_path)
        self.__train_dir = self.__image_path / train
        self.__val_dir = self.__image_path / val
        self.__test_dir = self.__image_path / test
        self.__output_dir = Path(output_dir)
        self.__filename = filename + ".pth"
        self.__epochs = epochs
        self.__unfreeze_classifier = unfreeze_classifier
        self.__unfreeze_specific_blocks = unfreeze_specific_blocks

        self.__train_dataloader, self.__val_dataloader , \
              self.__test_dataloader, \
                self.__class_names = self.__initializeDataHandler()
        
        self.__model = self.__initializeModel()
        self.__trainer = self.__initializeTrainer()
        self.__results = self.__trainer.run()
        
        self.__plots = self.__initializePLotCreator()
        self.__save_model_weights()

    def __initializeDataHandler(self):
        """This function creates DataHandler class instance."""
        datahandler = DataHandler(
            train_dir = str(self.__train_dir), 
            val_dir=str(self.__val_dir),
            test_dir = str(self.__test_dir)
            )
        train = datahandler.train_dataloader
        val = datahandler.val_dataloader
        test = datahandler.test_dataloader
        classes = datahandler.class_names
        return train, val, test, classes


    def __initializeModel(self):
        """This function creates ResNet50CustomModel class instance."""
        model = ResNet50CustomModel(
            unfreeze_classifier=self.__unfreeze_classifier,
            unfreeze_specific_blocks=self.__unfreeze_specific_blocks
        )
        return model

    def __initializeTrainer(self):
        """This function creates Trainer class instance."""
        trainer = Trainer(
            model=self.__model,
            train_dataloader=self.__train_dataloader,
            val_dataloader=self.__val_dataloader,
            epochs=self.__epochs   
        )
        return trainer
        
    def __initializePLotCreator(self):
        """This function creates PlotCreator class instance."""
        if self.__results:
            p = PlotCreator(self.__results)
            return p
        return None
    
    def __save_model_weights(self):
        if self.__model is not None:
            try:
                os.makedirs(self.__output_dir, exist_ok=True)
                model_save_path = self.__output_dir / self.__filename
                torch.save(self.__model.state_dict(), model_save_path)
            except AttributeError:
                 print(f"AttributeError: Could not save model weights. \
                       Ensure self.__model is a valid PyTorch nn.Module \
                       and has state_dict method.")
            except Exception as e:
                print(f"Error saving model weights to \
                       {model_save_path}: {e}")
        else:
            print("Model not initialized")

    @property
    def image_path(self):
        """This is a image path getter."""
        return self.__image_path
    
    @property
    def train_dir(self):
        """This is a train directory getter."""
        return self.__train_dir
        
    @property
    def test_dir(self):
        """This is a test directory getter."""
        return self.__test_dir
    
    @property
    def epochs(self):
        """This is a epochs number getter."""
        return self.__epochs
    
    @property
    def unfreeze_classifier(self):
        """This is a unfreeze classifier getter."""
        return self.__unfreeze_classifier

    @property
    def unfreeze_specific_blocks(self):
        """This is a getter for the list of bloacks that are unfreezed."""
        return self.__unfreeze_specific_blocks 

    @property
    def train_dataloader(self):
        """This is a train data loader getter."""
        return self.__train_dataloader
    
    @property
    def test_dataloader(self):
        """This is a test data loader getter."""
        return self.__test_dataloader
    
    @property
    def class_names(self):
        """This is a class names getter."""
        return self.__class_names
    
    @property
    def model(self):
        """This is a model getter."""
        return self.__model

    @property
    def trainer(self):
        """This is a trainer getter."""
        return self.__trainer
    
    @property
    def results(self):
        """This is a results getter."""
        return self.__results

    @property
    def plot_creator(self):
        """This is a plot creator getter."""
        return self.__plots
    
    @property
    def output_dir(self):
        """This is a output_dir getter."""
        return self.__output_dir
    
    @property
    def filename(self):
        """This is a file name getter."""
        return self.__filename