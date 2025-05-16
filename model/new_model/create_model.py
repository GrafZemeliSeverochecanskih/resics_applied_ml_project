from pathlib import Path
from utils.data_handler import DataHandler
from new_model.new_model import NewResNet50Model
from new_model.trainer import Trainer
from utils.mc_dropout import MonteCarloDropout
from utils.plot_creator import PlotCreator

class CreateNewModel:
    """This class interconnects different blocks n order to create and
    train new model from scratch.
    """
    def __init__(self,
                 image_path: str,
                 train: str = "train/",
                 test: str = "test/",
                 epochs: int = 10,
                 num_samples_mc = 10,
                 unfreeze_classifier=True,
                 unfreeze_specific_blocks=None,
                 monte_carlo_inference = True
                 ):
        """This is CreateNewModel class constructor."""
        self.__image_path = Path(image_path)
        self.__train_dir = self.__image_path / train
        self.__test_dir = self.__image_path / test
        self.__epochs = epochs
        self.__unfreeze_classifier = unfreeze_classifier
        self.__unfreeze_specific_blocks = unfreeze_specific_blocks
        self.__num_samples_mc = num_samples_mc
        self.__train_dataloader, self.__test_dataloader, \
            self.__class_names = self.__initializeDataHandler()
        self.__model = self.__initializeModel()
        self.__trainer = self.__initializeTrainer()
        self.__results = self.__trainer.run()
        
        if monte_carlo_inference:
            self.__mc = self.__initializeMonteCarloDropout()
            if self.__mc:
                self.__mc.run()
        self.__plots = self.__initializePLotCreator()

    def __initializeDataHandler(self):
        """This function creates DataHandler class instance."""
        datahandler = DataHandler(
            train_dir = str(self.__train_dir), 
            test_dir = str(self.__test_dir)
            )
        train = datahandler.train_dataloader
        test = datahandler.test_dataloader
        classes = datahandler.class_names
        return train, test, classes


    def __initializeModel(self):
        """This function creates NewResNet50Model class instance."""
        model = NewResNet50Model(
            unfreeze_classifier=self.__unfreeze_classifier,
            unfreeze_specific_blocks=self.__unfreeze_specific_blocks
        )
        return model

    def __initializeTrainer(self):
        """This function creates Trainer class instance."""
        trainer = Trainer(
            model=self.__model,
            train_dataloader=self.__train_dataloader,
            test_dataloader=self.__test_dataloader,
            epochs=self.__epochs   
        )
        return trainer

    def __initializeMonteCarloDropout(self):
        """This function creates MonteCarloDroput class instance."""
        mc = MonteCarloDropout(
            model=self.__model, 
            data_loader= self.__test_dataloader,
            num_samples= self.__num_samples_mc
            )
        return mc
        
    def __initializePLotCreator(self):
        """This function creates PlotCreator class instance."""
        if self.__results:
            p = PlotCreator(self.__results)
            return p
        
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
    def mc(self):
        """This is a mc dropout getter."""
        return self.__mc

    @property
    def plot_creator(self):
        """This is a plot creator getter."""
        return self.__plots
    